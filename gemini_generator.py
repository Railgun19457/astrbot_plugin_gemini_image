"""
Gemini Image Generation Module
封装 Gemini API 的图像生成功能，支持文生图和图生图
"""

from __future__ import annotations

import asyncio
import base64
import re
from io import BytesIO

import aiohttp
from PIL import Image

from astrbot.api import logger


class GeminiImageGenerator:
    """Gemini 图像生成器"""

    def __init__(
        self,
        api_keys: list[str],
        base_url: str,
        model: str,
        api_type: str = "gemini",
        timeout: int = 120,
        max_retry_attempts: int = 3,
        proxy: str | None = None,
        safety_settings: str = "BLOCK_NONE",
    ):
        self.api_keys = api_keys if api_keys else []
        self.current_key_index = 0
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_type = api_type
        self.timeout = timeout
        self.max_retry_attempts = max(1, min(max_retry_attempts, 10))
        self.proxy = proxy
        self.safety_settings = safety_settings
        self._session: aiohttp.ClientSession | None = None

    def _get_current_api_key(self) -> str:
        """获取当前使用的 API Key"""
        if not self.api_keys:
            return ""
        return self.api_keys[self.current_key_index % len(self.api_keys)]

    def _rotate_api_key(self):
        """切换到下一个 API Key"""
        if len(self.api_keys) > 1:
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
            logger.info(
                f"[Gemini Image] 切换到下一个 API Key (索引: {self.current_key_index})"
            )

    def _get_session(self) -> aiohttp.ClientSession:
        """获取或创建 aiohttp session"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close_session(self):
        """关闭 aiohttp session"""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    def _sync_convert_image_format(
        self, image_data: bytes, mime_type: str
    ) -> tuple[bytes, str]:
        """同步的图片格式转换逻辑（在线程池中执行）"""
        try:
            img = Image.open(BytesIO(image_data))

            # 处理透明图片
            if img.mode in ("RGBA", "LA", "P"):
                background = Image.new("RGB", img.size, (255, 255, 255))
                if img.mode == "P":
                    img = img.convert("RGBA")
                elif img.mode == "LA":
                    img = img.convert("RGBA")

                # 此时 img.mode 一定是 RGBA，使用第4个通道作为 alpha mask
                background.paste(img, mask=img.split()[3])
                img = background

            # 转换为 JPEG
            output = BytesIO()
            img.save(output, format="JPEG", quality=95)

            logger.debug("[Gemini Image] 图片格式转换成功")
            return output.getvalue(), "image/jpeg"

        except Exception as e:
            logger.error(f"[Gemini Image] 图片格式转换失败: {e}")
            return image_data, mime_type

    def _detect_mime_type(self, data: bytes) -> str:
        """检测图片 MIME 类型"""
        mime = "application/octet-stream"
        if data.startswith(b"\xff\xd8"):
            mime = "image/jpeg"
        elif data.startswith(b"\x89PNG\r\n\x1a\n"):
            mime = "image/png"
        elif data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
            mime = "image/gif"
        elif data.startswith(b"RIFF") and data[8:12] == b"WEBP":
            mime = "image/webp"
        elif len(data) > 12 and data[4:8] == b"ftyp":
            brand = data[8:12]
            if brand in (b"heic", b"heix", b"heim", b"heis"):
                mime = "image/heic"
            elif brand in (b"mif1", b"msf1", b"heif"):
                mime = "image/heif"

        logger.debug(f"[Gemini Image] Detect MIME: {mime}")
        return mime

    async def _convert_image_format(
        self, image_data: bytes, mime_type: str
    ) -> tuple[bytes, str]:
        """转换不支持的图片格式为 JPEG"""
        real_mime = self._detect_mime_type(image_data)

        supported_formats = [
            "image/png",
            "image/jpeg",
            "image/webp",
            "image/heic",
            "image/heif",
        ]

        if real_mime in supported_formats:
            return image_data, real_mime

        logger.info(f"[Gemini Image] 转换图片格式: {mime_type} -> image/jpeg")
        return await asyncio.to_thread(
            self._sync_convert_image_format, image_data, mime_type
        )

    async def generate_image(
        self,
        prompt: str,
        images_data: list[tuple[bytes, str]] | None = None,
        aspect_ratio: str | None = "1:1",
        image_size: str | None = None,
        task_id: str | None = None,
    ) -> tuple[list[bytes] | None, str | None]:
        """生成图片"""
        logger.debug(
            f"[Gemini Image] generate_image params - Model: {self.model}, Aspect: {aspect_ratio}, Size: {image_size}, TaskID: {task_id}"
        )
        if not self.api_keys:
            return None, "未配置 API Key"

        prefix = f"[{task_id}] " if task_id else ""

        # 转换所有图片格式
        converted_images = []
        if images_data:
            for image_data, mime_type in images_data:
                converted_data, converted_mime = await self._convert_image_format(
                    image_data, mime_type
                )
                converted_images.append((converted_data, converted_mime))

        last_error = "未配置 API Key"

        for attempt in range(self.max_retry_attempts):
            if attempt > 0:
                logger.info(
                    f"[Gemini Image] {prefix}重试生成 (第 {attempt + 1}/{self.max_retry_attempts} 次)"
                )

            result = await self._try_generate_with_current_key(
                prompt, converted_images, aspect_ratio, image_size, task_id
            )

            if result[0] is not None:
                return result

            last_error = result[1] or "生成失败"

            if attempt < self.max_retry_attempts - 1:
                if len(self.api_keys) > 1:
                    self._rotate_api_key()

                # 如果是单 Key，或者已经轮询了一圈 Key，则进行退避等待
                if (attempt + 1) % len(self.api_keys) == 0:
                    round_index = (attempt + 1) // len(self.api_keys) - 1
                    wait_time = min(2**round_index, 10)
                    await asyncio.sleep(wait_time)

        return None, f"重试失败: {last_error}"

    async def _try_generate_with_current_key(
        self,
        prompt: str,
        images_data: list[tuple[bytes, str]],
        aspect_ratio: str | None,
        image_size: str | None,
        task_id: str | None = None,
    ) -> tuple[list[bytes] | None, str | None]:
        """使用当前 API Key 尝试生成图片"""
        if self.api_type == "gemini":
            return await self._generate_gemini(
                prompt, images_data, aspect_ratio, image_size, task_id
            )
        elif self.api_type == "zai":
            return await self._generate_zai(
                prompt, images_data, aspect_ratio, image_size, task_id
            )
        else:
            return await self._generate_openai(
                prompt, images_data, aspect_ratio, image_size, task_id
            )

    def _log_payload(self, payload: dict, provider: str):
        """记录请求负载，隐藏过长的 Base64 数据"""
        try:

            def truncate(obj):
                if isinstance(obj, dict):
                    return {k: truncate(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [truncate(i) for i in obj]
                elif isinstance(obj, str):
                    if len(obj) > 200:
                        return obj[:50] + f"...({len(obj)} chars)..." + obj[-20:]
                    return obj
                else:
                    return obj

            log_payload = truncate(payload)
            logger.debug(f"[Gemini Image] {provider} Payload: {log_payload}")
        except Exception:
            pass

    # ==================== OpenAI Implementation ====================

    async def _generate_openai(
        self,
        prompt: str,
        images_data: list[tuple[bytes, str]],
        aspect_ratio: str | None,
        image_size: str | None,
        task_id: str | None = None,
    ) -> tuple[list[bytes] | None, str | None]:
        """使用 OpenAI 兼容格式 API 生成图片 (Chat Completions)"""
        prefix = f"[{task_id}] " if task_id else ""

        try:
            # 统一使用 Chat Completions 接口
            payload = self._build_openai_chat_payload(
                prompt, images_data, aspect_ratio, image_size
            )
            endpoint_type = "v1/chat/completions"

            session = self._get_session()
            response_data = await self._make_openai_request(
                session, payload, endpoint_type, task_id, provider="OpenAI"
            )

            if response_data is None:
                return None, "API 请求失败"

            images = await self._extract_openai_chat_image(response_data)
            if images:
                return images, None

            # 尝试提取文本错误信息
            if "choices" in response_data and response_data["choices"]:
                content = response_data["choices"][0].get("message", {}).get("content")
                if content and isinstance(content, str):
                    return None, f"未生成图片，API返回文本: {content[:100]}"

            return None, "响应中未找到图片数据"

        except asyncio.TimeoutError:
            return None, "生成超时"
        except Exception as e:
            logger.error(f"[Gemini Image] {prefix}OpenAI 生成失败: {e}")
            return None, f"生成失败: {str(e)}"

    def _build_openai_chat_payload(
        self,
        prompt: str,
        images_data: list[tuple[bytes, str]] | None,
        aspect_ratio: str | None,
        image_size: str | None,
    ) -> dict:
        """构建 OpenAI Chat Completions 请求负载"""

        message_content = [{"type": "text", "text": f"Generate an image: {prompt}"}]

        # 处理参考图
        if images_data:
            for image_bytes, mime_type in images_data:
                b64_data = base64.b64encode(image_bytes).decode("utf-8")
                image_url = f"data:{mime_type};base64,{b64_data}"
                message_content.append(
                    {"type": "image_url", "image_url": {"url": image_url}}
                )

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": message_content}],
            "modalities": ["image", "text"],
            "stream": False,
        }

        # image_config
        image_config = {}
        generation_config = {}

        if aspect_ratio and not images_data:
            image_config["aspectRatio"] = aspect_ratio

        if image_size:
            image_config["imageSize"] = image_size

        if image_config:
            generation_config["imageConfig"] = image_config

        payload["generationConfig"] = generation_config

        self._log_payload(payload, "OpenAI Chat")

        return payload

    async def _make_openai_request(
        self,
        session: aiohttp.ClientSession,
        payload: dict,
        endpoint_type: str,
        task_id: str | None,
        provider: str = "OpenAI",
    ) -> dict | None:
        prefix = f"[{task_id}] " if task_id else ""

        # 构建 URL
        url = f"{self.base_url}/{endpoint_type}"

        api_key = self._get_current_api_key()
        masked_key = api_key[:4] + "****" + api_key[-4:] if len(api_key) > 8 else "****"
        logger.debug(f"[Gemini Image] {prefix}Request URL: {url}, Key: {masked_key}")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        async with session.post(
            url,
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=self.timeout),
            proxy=self.proxy,
        ) as response:
            logger.debug(
                f"[Gemini Image] {prefix}API Response Status: {response.status}"
            )
            if response.status != 200:
                error_text = await response.text()
                error_preview = (
                    error_text[:200] + "..." if len(error_text) > 200 else error_text
                )
                logger.error(
                    f"[Gemini Image] {prefix}{provider} API 错误: {response.status} - {error_preview}"
                )
                return None
            return await response.json()

    async def _download_image_from_url(self, url: str) -> bytes | None:
        """从 URL 下载图片"""
        try:
            session = self._get_session()
            async with session.get(url, timeout=30) as response:
                if response.status == 200:
                    return await response.read()
                else:
                    logger.error(
                        f"[Gemini Image] 下载图片失败: {response.status} - {url}"
                    )
                    return None
        except Exception as e:
            logger.error(f"[Gemini Image] 下载图片异常: {e}")
            return None

    async def _extract_openai_chat_image(
        self, response_data: dict
    ) -> list[bytes] | None:
        """从 OpenAI Chat 响应中提取图片"""
        images = []

        # 0. 检查标准 OpenAI DALL-E 格式 (data 字段)
        if "data" in response_data and isinstance(response_data["data"], list):
            for item in response_data["data"]:
                if isinstance(item, dict):
                    b64_json = item.get("b64_json")
                    url = item.get("url")
                    if b64_json:
                        try:
                            images.append(base64.b64decode(b64_json))
                        except Exception:
                            pass
                    elif url:
                        if url.startswith("http"):
                            img_data = await self._download_image_from_url(url)
                            if img_data:
                                images.append(img_data)
                        else:
                            img_data = self._decode_image_url(url)
                            if img_data:
                                images.append(img_data)

        if "choices" in response_data and response_data["choices"]:
            choice = response_data["choices"][0]
            message = choice.get("message", {})
            content = message.get("content")

            # 1. 检查 message.content 为字符串的情况 (Markdown 图片 / Data URI)
            if isinstance(content, str):
                # 匹配 markdown 图片语法 ![...](url)
                markdown_matches = re.findall(r"!\[.*?\]\((.*?)\)", content)
                for url in markdown_matches:
                    if url.startswith("http"):
                        img_data = await self._download_image_from_url(url)
                        if img_data:
                            images.append(img_data)
                    else:
                        img_data = self._decode_image_url(url)
                        if img_data:
                            images.append(img_data)

                # 匹配纯文本中的 Data URI（排除 markdown 语法中已匹配的部分）
                # 先移除 markdown 图片语法，避免重复解析
                content_without_markdown = re.sub(r"!\[.*?\]\(.*?\)", "", content)
                pattern = re.compile(
                    r"data\s*:\s*image/([a-zA-Z0-9.+-]+)\s*;\s*base64\s*,\s*([-A-Za-z0-9+/=_\s]+)",
                    flags=re.IGNORECASE,
                )
                data_uri_matches = pattern.findall(content_without_markdown)
                for _, b64_str in data_uri_matches:
                    try:
                        images.append(base64.b64decode(b64_str))
                    except Exception:
                        pass

            # 2. 检查 message.content 中的 image_url (Gemini 风格)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        image_url = part.get("image_url", {}).get("url")
                        if image_url:
                            if image_url.startswith("http"):
                                img_data = await self._download_image_from_url(
                                    image_url
                                )
                                if img_data:
                                    images.append(img_data)
                            else:
                                img_data = self._decode_image_url(image_url)
                                if img_data:
                                    images.append(img_data)

            # 3. 检查 message.images 字段 (非标准字段，部分实现可能使用)
            if message.get("images"):
                for img_item in message["images"]:
                    url = None
                    if isinstance(img_item, dict):
                        url = img_item.get("url") or img_item.get("image_url", {}).get(
                            "url"
                        )
                    elif isinstance(img_item, str):
                        url = img_item

                    if url:
                        if url.startswith("http"):
                            img_data = await self._download_image_from_url(url)
                            if img_data:
                                images.append(img_data)
                        else:
                            img_data = self._decode_image_url(url)
                            if img_data:
                                images.append(img_data)

        return images if images else None

    def _decode_image_url(self, url: str) -> bytes | None:
        """解码 data:image/...;base64,... URL"""
        if url.startswith("data:image/") and ";base64," in url:
            try:
                _, _, data_part = url.partition(";base64,")
                return base64.b64decode(data_part)
            except Exception as e:
                logger.error(f"[Gemini Image] Base64 解码失败: {e}")
                return None
        return None

    # ==================== Zai Implementation ====================

    async def _generate_zai(
        self,
        prompt: str,
        images_data: list[tuple[bytes, str]],
        aspect_ratio: str | None,
        image_size: str | None,
        task_id: str | None = None,
    ) -> tuple[list[bytes] | None, str | None]:
        """使用 Zai 格式 API 生成图片"""
        prefix = f"[{task_id}] " if task_id else ""

        try:
            payload = self._build_zai_payload(
                prompt, images_data, aspect_ratio, image_size
            )

            # Zai 使用 OpenAI 兼容的 endpoint
            endpoint_type = "v1/chat/completions"
            session = self._get_session()

            # 复用 OpenAI 的请求逻辑
            response_data = await self._make_openai_request(
                session, payload, endpoint_type, task_id, provider="Zai"
            )

            if response_data is None:
                return None, "API 请求失败"

            # 尝试使用 OpenAI 兼容的提取逻辑
            images = await self._extract_openai_chat_image(response_data)
            if images:
                return images, None

            # 尝试提取文本错误信息
            if "choices" in response_data and response_data["choices"]:
                content = response_data["choices"][0].get("message", {}).get("content")
                if content and isinstance(content, str):
                    return None, f"未生成图片，API返回文本: {content[:100]}"

            return None, "响应中未找到图片数据"

        except asyncio.TimeoutError:
            return None, "生成超时"
        except Exception as e:
            logger.error(f"[Gemini Image] {prefix}Zai 生成失败: {e}")
            return None, f"生成失败: {str(e)}"

    def _build_zai_payload(
        self,
        prompt: str,
        images_data: list[tuple[bytes, str]] | None,
        aspect_ratio: str | None,
        image_size: str | None,
    ) -> dict:
        """构建 Zai 请求负载 (OpenAI 格式 + params)"""

        # Zai 不需要 "Generate an image: " 前缀，直接使用 prompt
        message_content = [{"type": "text", "text": prompt}]

        # 处理参考图 (OpenAI Vision 格式)
        if images_data:
            for image_bytes, mime_type in images_data:
                b64_data = base64.b64encode(image_bytes).decode("utf-8")
                image_url = f"data:{mime_type};base64,{b64_data}"
                message_content.append(
                    {"type": "image_url", "image_url": {"url": image_url}}
                )

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": message_content}],
            "stream": False,
        }

        # Zai 特有参数
        params = {}
        if aspect_ratio and not images_data:
            params["image_aspect_ratio"] = aspect_ratio
        if image_size:
            params["image_resolution"] = image_size

        if params:
            payload["params"] = params

        self._log_payload(payload, "Zai")
        return payload

    async def _make_zai_request(
        self,
        session: aiohttp.ClientSession,
        payload: dict,
        task_id: str | None,
    ) -> dict | None:
        # Deprecated: 使用 _make_openai_request 代替
        pass

    # ==================== Gemini Implementation ====================

    async def _generate_gemini(
        self,
        prompt: str,
        images_data: list[tuple[bytes, str]],
        aspect_ratio: str | None,
        image_size: str | None,
        task_id: str | None = None,
    ) -> tuple[list[bytes] | None, str | None]:
        """使用 Gemini 格式 API 生成图片"""
        prefix = f"[{task_id}] " if task_id else ""

        try:
            payload = self._build_gemini_payload(
                prompt, images_data, aspect_ratio, image_size
            )

            session = self._get_session()
            response_data = await self._make_gemini_request(session, payload, task_id)
            if response_data is None:
                return None, "API 请求失败"

            result_image_data = self._extract_gemini_image(response_data, task_id)
            if result_image_data:
                return result_image_data, None

            return None, "响应中未找到图片数据"

        except asyncio.TimeoutError:
            return None, "生成超时"
        except Exception as e:
            logger.error(f"[Gemini Image] {prefix}Gemini 生成失败: {e}")
            return None, f"生成失败: {str(e)}"

    def _build_gemini_payload(
        self,
        prompt: str,
        images_data: list[tuple[bytes, str]],
        aspect_ratio: str | None,
        image_size: str | None,
    ) -> dict:
        generation_config = {"responseModalities": ["IMAGE"]}
        image_config = {}
        # 如果有参考图，则不传 aspect_ratio，使用参考图的比例
        if aspect_ratio and not images_data:
            image_config["aspectRatio"] = aspect_ratio

        # imageSize 仅 gemini-3-pro-image-preview 支持
        if image_size and "gemini-3" in self.model.lower():
            image_config["imageSize"] = image_size

        if image_config:
            generation_config["imageConfig"] = image_config

        safety_settings = []
        if self.safety_settings:
            for category in [
                "HARM_CATEGORY_HARASSMENT",
                "HARM_CATEGORY_HATE_SPEECH",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "HARM_CATEGORY_DANGEROUS_CONTENT",
                "HARM_CATEGORY_CIVIC_INTEGRITY",
            ]:
                safety_settings.append(
                    {"category": category, "threshold": self.safety_settings}
                )

        parts = [{"text": prompt}]

        if images_data:
            for image_data, mime_type in images_data:
                encoded_data = base64.b64encode(image_data).decode("utf-8")
                parts.append(
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": encoded_data,
                        }
                    }
                )

        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": generation_config,
            "safetySettings": safety_settings,
        }
        self._log_payload(payload, "Gemini")
        return payload

    async def _make_gemini_request(
        self, session: aiohttp.ClientSession, payload: dict, task_id: str | None = None
    ) -> dict | None:
        prefix = f"[{task_id}] " if task_id else ""
        url = f"{self.base_url}/v1beta/models/{self.model}:generateContent"

        api_key = self._get_current_api_key()
        masked_key = api_key[:4] + "****" + api_key[-4:] if len(api_key) > 8 else "****"
        logger.debug(f"[Gemini Image] {prefix}Request URL: {url}, Key: {masked_key}")

        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        }

        async with session.post(
            url,
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=self.timeout),
            proxy=self.proxy,
        ) as response:
            logger.debug(
                f"[Gemini Image] {prefix}API Response Status: {response.status}"
            )
            if response.status != 200:
                error_text = await response.text()
                # 截断过长的错误信息
                error_preview = (
                    error_text[:200] + "..." if len(error_text) > 200 else error_text
                )
                logger.error(
                    f"[Gemini Image] {prefix}Gemini API 错误: {response.status} - {error_preview}"
                )
                if response.status in [401, 403, 429]:
                    return None
                return None
            return await response.json()

    def _extract_gemini_image(
        self, response: dict, task_id: str | None = None
    ) -> list[bytes] | None:
        """从 API 响应中提取图片数据"""
        prefix = f"[{task_id}] " if task_id else ""
        try:
            candidates = response.get("candidates", [])
            logger.debug(f"[Gemini Image] {prefix}Candidates count: {len(candidates)}")
            if not candidates:
                return None

            parts = candidates[0].get("content", {}).get("parts", [])
            images = []
            for part in parts:
                inline_data = part.get("inline_data") or part.get("inlineData")
                if inline_data:
                    data = inline_data.get("data")
                    if data:
                        images.append(base64.b64decode(data))

            return images if images else None

        except Exception as e:
            logger.error(f"[Gemini Image] 解析响应失败: {e}")
            return None
