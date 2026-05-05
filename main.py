"""
Gemini Image Generation Plugin
使用 Gemini 系列模型进行图像生成的插件
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from collections.abc import Coroutine
from typing import Any

from pydantic import Field
from pydantic.dataclasses import dataclass as pydantic_dataclass

import astrbot.api.message_components as Comp
from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.event.filter import EventMessageType
from astrbot.api.star import Context, Star
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.agent.tool import FunctionTool, ToolExecResult
from astrbot.core.astr_agent_context import AstrAgentContext
from astrbot.core.config.astrbot_config import AstrBotConfig
from astrbot.core.utils.io import download_image_by_url, save_temp_img

from .gemini_generator import GeminiImageGenerator


@pydantic_dataclass
class GeminiImageGenerationTool(FunctionTool[AstrAgentContext]):
    """统一的图像生成工具，支持文生图和图生图"""

    name: str = "gemini_generate_image"
    description: str = "使用 Gemini 模型生成或修改图片(需权限验证，非授权用户请勿调用)"
    parameters: dict = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "生图时使用的提示词(直接将用户发送的内容原样传递给模型)",
                },
                "aspect_ratio": {
                    "type": "string",
                    "description": "图片宽高比",
                    "enum": [
                        "1:1",
                        "2:3",
                        "3:2",
                        "3:4",
                        "4:3",
                        "4:5",
                        "5:4",
                        "9:16",
                        "16:9",
                        "21:9",
                    ],
                },
                "resolution": {
                    "type": "string",
                    "description": "图片分辨率，仅 gemini-3-pro-image-preview(nano banana pro) 模型支持",
                    "enum": ["1K", "2K", "4K"],
                },
                "avatar_references": {
                    "type": "array",
                    "description": "需要作为参考的用户头像列表。支持: 'self'(机器人头像)、'sender'(发送者头像)、或具体的QQ号",
                    "items": {"type": "string"},
                },
            },
            "required": ["prompt"],
        }
    )

    plugin: object | None = None

    async def call(
        self, context: ContextWrapper[AstrAgentContext], **kwargs
    ) -> ToolExecResult:
        if not (prompt := kwargs.get("prompt", "")):
            return "请提供图片生成的提示词"

        plugin = self.plugin
        if not plugin:
            return "❌ 插件未正确初始化 (Plugin instance missing)"

        event = None
        if hasattr(context, "context") and isinstance(
            context.context, AstrAgentContext
        ):
            event = context.context.event
        elif isinstance(context, dict):
            event = context.get("event")

        if not event:
            logger.warning(
                f"[Gemini Image] Tool call context missing event. Context type: {type(context)}"
            )
            return "❌ 无法获取当前消息上下文"

        # Enhanced User ID Extraction & Debugging
        user_id = event.get_sender_id()
        if not user_id and event.message_obj and event.message_obj.sender:
            user_id = event.message_obj.sender.user_id
        if not user_id:
            user_id = event.unified_msg_origin

        user_id = str(user_id).strip()
        group_id = event.message_obj.group_id or ""

        # DEBUG: Log the values used for permission check
        logger.info(
            f"[Gemini Permission] Tool Call - User: {user_id}, Group: {group_id}"
        )
        if hasattr(plugin, "perm_users"):
            logger.info(
                f"[Gemini Permission] Config - Mode: {plugin.perm_mode}, Users: {plugin.perm_users}"
            )

        # Permission Check
        if hasattr(plugin, "_check_permission"):
            if not plugin._check_permission(user_id, group_id):
                logger.warning(f"[Gemini Permission] DENIED for User: {user_id}")
                # SYSTEM STOP INSTRUCTION
                if plugin.perm_silent:
                    return "SYSTEM_NOTIFICATION: Permission denied. User unauthorized. EXECUTION STOPPED. Do NOT generate image. Do NOT reply to the user. END."
                else:
                    return f"SYSTEM_NOTIFICATION: Permission denied. EXECUTION STOPPED. Reply to user: '{plugin.perm_no_permission_reply}'"
            else:
                logger.info(f"[Gemini Permission] ALLOWED for User: {user_id}")
        else:
            logger.error(
                "[Gemini Permission] Plugin instance missing _check_permission method"
            )

        if not plugin.generator.api_keys:
            return "❌ 未配置 API Key，无法生成图片"

        # 获取参考图片
        images_data = await plugin._get_reference_images_for_tool(event)

        # 处理头像引用参数
        avatar_references = kwargs.get("avatar_references", [])
        if avatar_references and isinstance(avatar_references, list):
            for ref in avatar_references:
                if not isinstance(ref, str):
                    continue

                ref = ref.strip().lower()
                user_id = None

                if ref == "self":
                    # 获取机器人自己的头像
                    user_id = str(event.get_self_id())
                elif ref == "sender":
                    # 获取发送者的头像
                    user_id = str(event.get_sender_id() or event.unified_msg_origin)
                else:
                    # 作为QQ号处理
                    user_id = ref

                if user_id:
                    avatar_data = await plugin.get_avatar(user_id)
                    if avatar_data:
                        images_data.append((avatar_data, "image/jpeg"))
                        logger.info(
                            f"[Gemini Image] 已添加用户 {user_id} 的头像作为参考图片"
                        )
                    else:
                        logger.warning(f"[Gemini Image] 无法获取用户 {user_id} 的头像")

        # 生成任务 ID
        task_id = hashlib.md5(
            f"{time.time()}{event.unified_msg_origin}".encode()
        ).hexdigest()[:8]

        # 记录任务摘要
        res = kwargs.get("resolution", plugin.default_resolution)
        ar = kwargs.get("aspect_ratio", plugin.default_aspect_ratio)
        img_count = len(images_data) if images_data else 0
        logger.info(
            f"[Gemini Image] 任务摘要 [{task_id}] - 提示词: {prompt} | 预设: 无 | 参考图: {img_count}张 | 分辨率: {res} | 比例: {ar}"
        )

        plugin.create_background_task(
            plugin._generate_and_send_image_async(
                prompt=prompt,
                images_data=images_data or None,
                unified_msg_origin=event.unified_msg_origin,
                aspect_ratio=ar,
                resolution=res,
                task_id=task_id,
            )
        )

        mode = "图生图" if images_data else "文生图"
        return f"已启动{mode}任务"


class GeminiImagePlugin(Star):
    """Gemini 图像生成插件"""

    # 配置验证常量
    DEFAULT_MAX_CONCURRENT_GENERATIONS = 3
    MAX_CONCURRENT_GENERATIONS = 10

    # 可用模型列表
    AVAILABLE_MODELS = [
        "gemini-2.0-flash-exp-image-generation",
        "gemini-2.5-flash-image",
        "gemini-2.5-flash-image-preview",
        "gemini-3-pro-image-preview",
    ]

    def __init__(self, context: Context, config: AstrBotConfig | None = None):
        super().__init__(context)
        self.context = context
        self.config = config or AstrBotConfig()

        # 读取配置
        self._load_config()

        # 初始化生成器
        self.generator = GeminiImageGenerator(
            api_keys=self.api_keys,
            base_url=self.base_url,
            model=self.model,
            api_type=self.api_type,
            timeout=self.timeout,
            max_retry_attempts=self.max_retry_attempts,
            proxy=self.proxy,
            safety_settings=self.safety_settings,
        )

        self.background_tasks: set[asyncio.Task] = set()
        self._generation_semaphore = asyncio.Semaphore(self.max_concurrent_generations)

        # 频率限制 {user_id: [timestamp, ...]}
        self.user_request_timestamps: dict[str, list[float]] = {}

        # 注册工具到 LLM
        if self.enable_llm_tool:
            self.context.add_llm_tools(GeminiImageGenerationTool(plugin=self))
            logger.info("[Gemini Image] 已注册图像生成工具（支持头像引用）")

        logger.info(f"[Gemini Image] 插件已加载，使用模型: {self.model}")

    def _load_config(self):
        """加载配置"""
        # 读取分组配置
        api_config = self.config.get("api_config", {})
        generate_config = self.config.get("generate_config", {})

        # API 配置组
        self.api_type = api_config.get("api_type", "gemini")
        use_system_provider = api_config.get("use_system_provider", True)
        provider_id = (api_config.get("provider_id", "") or "").strip()

        if (
            use_system_provider
            and provider_id
            and self._load_provider_config(provider_id)
        ):
            pass
        else:
            if use_system_provider and not provider_id:
                logger.warning("[Gemini Image] 未配置提供商 ID，将使用插件配置")
            self._load_default_config()

        self.model = self._load_model_config()
        self.proxy = api_config.get("proxy", "") or None

        # 生图配置组
        self.timeout = generate_config.get("timeout", 300)
        self.default_aspect_ratio = generate_config.get("default_aspect_ratio", "1:1")
        self.default_resolution = generate_config.get("default_resolution", "1K")
        self.max_retry_attempts = generate_config.get("max_retry_attempts", 3)
        self.safety_settings = generate_config.get("safety_settings", "BLOCK_NONE")
        self.max_image_size_mb = generate_config.get("max_image_size_mb", 10)
        self.enable_rate_limit = generate_config.get("enable_rate_limit", True)
        self.max_requests_per_minute = generate_config.get("max_requests_per_minute", 3)
        self.max_requests_per_hour = generate_config.get("max_requests_per_hour", 30)
        self.max_requests_per_day = generate_config.get("max_requests_per_day", 100)

        # 验证并发配置
        max_concurrent = generate_config.get(
            "max_concurrent_generations", self.DEFAULT_MAX_CONCURRENT_GENERATIONS
        )
        self.max_concurrent_generations = min(
            max(1, max_concurrent), self.MAX_CONCURRENT_GENERATIONS
        )

        # 顶层配置
        self.enable_llm_tool = self.config.get("enable_llm_tool", True)
        self.presets = self._load_presets()

        # 权限配置
        perm_conf = self.config.get("permission_config", {})
        self.perm_mode = perm_conf.get("mode", "disable")
        self.perm_users = set(perm_conf.get("users", []))
        self.perm_groups = set(perm_conf.get("groups", []))
        self.perm_no_permission_reply = perm_conf.get(
            "no_permission_reply", "❌ 您没有权限使用此功能"
        )
        self.perm_silent = perm_conf.get("silent_on_no_permission", False)

    def _check_permission(self, user_id: str, group_id: str = "") -> bool:
        """检查权限"""
        # 实时读取配置
        perm_conf = self.config.get("permission_config", {})
        mode = perm_conf.get("mode", "disable").strip()

        if mode == "disable":
            return True

        user_id = str(user_id).strip()
        group_id = str(group_id).strip()

        # 统一转为字符串集合进行比对，去除空格
        limit_users = {str(u).strip() for u in perm_conf.get("users", [])}
        limit_groups = {str(g).strip() for g in perm_conf.get("groups", [])}

        logger.info(
            f"[Gemini Image] Perm Check: mode={mode}, user={user_id}, lists={limit_users}, groups={limit_groups}, group_id={group_id}"
        )

        if mode == "blacklist":
            # 黑名单模式: 在名单内则禁止
            if user_id in limit_users:
                return False
            if group_id and group_id in limit_groups:
                return False
            return True

        elif mode == "whitelist":
            # 白名单模式: 在名单内才允许
            if user_id in limit_users:
                return True
            if group_id and group_id in limit_groups:
                return True
            # 不在白名单中 -> 禁止
            return False

        return True

    def _clean_base_url(self, url: str) -> str:
        """清洗 Base URL"""
        if not url:
            return ""
        url = url.rstrip("/")
        # 移除 /v1 及其后的所有内容 (包括 /v1beta, /v1/chat 等)
        if "/v1" in url:
            url = url.split("/v1", 1)[0]
        return url.rstrip("/")

    def _load_provider_config(self, provider_id: str) -> bool:
        """从系统提供商加载配置"""
        provider = self.context.get_provider_by_id(provider_id)
        if not provider:
            logger.warning(f"[Gemini Image] 未找到提供商 {provider_id}，将使用插件配置")
            return False

        provider_config = getattr(provider, "provider_config", {}) or {}

        # 提取 keys
        api_keys = []
        for key_field in ["key", "keys", "api_key", "access_token"]:
            if keys := provider_config.get(key_field):
                api_keys = [keys] if isinstance(keys, str) else [k for k in keys if k]
                break

        # 提取 base_url
        api_base = (
            getattr(provider, "api_base", None)
            or provider_config.get("api_base")
            or provider_config.get("api_base_url")
        )

        if not api_keys:
            logger.warning(f"[Gemini Image] 提供商 {provider_id} 未提供可用的 API Key")
            return False

        self.api_keys = api_keys
        self.base_url = self._clean_base_url(
            api_base or "https://generativelanguage.googleapis.com"
        )

        logger.info(f"[Gemini Image] 使用系统提供商: {provider_id}")
        return True

    def _load_model_config(self) -> str:
        """加载模型配置"""
        api_config = self.config.get("api_config", {})
        model = api_config.get("model", "gemini-2.0-flash-exp-image-generation")
        if model != "自定义模型":
            return model
        return (
            api_config.get("custom_model", "").strip()
            or "gemini-2.0-flash-exp-image-generation"
        )

    def _load_presets(self) -> dict[str, str]:
        """加载预设提示词配置"""
        presets_config = self.config.get("presets", [])
        presets_dict = {}

        if not isinstance(presets_config, list):
            return presets_dict

        for preset_str in presets_config:
            if isinstance(preset_str, str) and ":" in preset_str:
                name, prompt = preset_str.split(":", 1)
                if name.strip() and prompt.strip():
                    presets_dict[name.strip()] = prompt.strip()

        return presets_dict

    def _load_default_config(self):
        """加载默认配置"""
        api_config = self.config.get("api_config", {})
        api_key = api_config.get("api_key", "")
        self.api_keys = (
            [k for k in api_key if k]
            if isinstance(api_key, list)
            else [api_key]
            if api_key
            else []
        )
        default_base = "https://generativelanguage.googleapis.com"
        self.base_url = self._clean_base_url(api_config.get("base_url", default_base))

    def _check_rate_limit(self, user_id: str) -> tuple[bool, str]:
        """检查用户请求频率是否超限"""
        if not getattr(self, "enable_rate_limit", True):
            return True, ""

        now = time.time()
        timestamps = self.user_request_timestamps.setdefault(user_id, [])

        # 保留最近一天的记录即可（24小时 = 86400秒）
        valid_timestamps = [t for t in timestamps if now - t < 86400]
        self.user_request_timestamps[user_id] = valid_timestamps

        # 统计分钟、小时、天的请求数
        count_minute = sum(1 for t in valid_timestamps if now - t < 60)
        count_hour = sum(1 for t in valid_timestamps if now - t < 3600)
        count_day = len(valid_timestamps)

        if count_minute >= getattr(self, "max_requests_per_minute", 3):
            return (
                False,
                f"❌ 请求过于频繁，请稍后再试 (每分钟限 {self.max_requests_per_minute} 次)",
            )
        if count_hour >= getattr(self, "max_requests_per_hour", 30):
            return (
                False,
                f"❌ 请求过于频繁，请稍后再试 (每小时限 {self.max_requests_per_hour} 次)",
            )
        if count_day >= getattr(self, "max_requests_per_day", 100):
            return (
                False,
                f"❌ 请求过于频繁，请明天再试 (每天限 {self.max_requests_per_day} 次)",
            )

        valid_timestamps.append(now)
        return True, ""

    @filter.command("生图")
    async def generate_image_command(self, event: AstrMessageEvent):
        """生成图片指令"""
        user_id = str(event.get_sender_id() or event.unified_msg_origin)
        group_id = event.message_obj.group_id or ""

        if not self._check_permission(user_id, group_id):
            # 权限不足
            if not self.perm_silent:
                yield event.plain_result(self.perm_no_permission_reply)
            return

        is_allowed, rate_msg = self._check_rate_limit(user_id)
        if not is_allowed:
            yield event.plain_result(rate_msg)
            return

        masked_uid = (
            user_id[:4] + "****" + user_id[-4:] if len(user_id) > 8 else user_id
        )

        user_input = (event.message_str or "").strip()
        logger.info(
            f"[Gemini Image] 收到生图指令 - 用户: {masked_uid}, 原始输入: {user_input}"
        )

        # 移除指令前缀
        cmd_parts = user_input.split(maxsplit=1)
        if not cmd_parts:
            return

        # 如果只有指令本身，且没有参数
        prompt = ""
        if len(cmd_parts) > 1:
            prompt = cmd_parts[1].strip()

        # 默认参数
        aspect_ratio = self.default_aspect_ratio
        resolution = self.default_resolution

        # 检查是否使用了预设
        matched_preset = None
        extra_content = ""

        if prompt:
            # 分割第一部分作为潜在的预设名称
            parts = prompt.split(maxsplit=1)
            first_token = parts[0]
            rest_token = parts[1] if len(parts) > 1 else ""

            # 检查是否匹配预设
            if first_token in self.presets:
                matched_preset = first_token
                extra_content = rest_token
            else:
                # 大小写不敏感匹配
                for name in self.presets:
                    if name.lower() == first_token.lower():
                        matched_preset = name
                        extra_content = rest_token
                        break

        if matched_preset:
            logger.info(f"[Gemini Image] 命中预设: {matched_preset}")
            preset_content = self.presets[matched_preset]

            # 尝试解析 JSON 格式的预设
            try:
                if preset_content.strip().startswith("{"):
                    preset_data = json.loads(preset_content)
                    if isinstance(preset_data, dict):
                        prompt = preset_data.get("prompt", "")
                        aspect_ratio = preset_data.get("aspect_ratio", aspect_ratio)
                        resolution = preset_data.get("resolution", resolution)
                    else:
                        prompt = preset_content
                else:
                    prompt = preset_content
            except json.JSONDecodeError:
                prompt = preset_content

            # 如果有额外内容，追加到提示词后
            if extra_content:
                prompt = f"{prompt} {extra_content}"

        if not prompt:
            yield event.plain_result("❌ 请提供图片生成的提示词或预设名称！")
            return

        # 获取参考图片
        images_data = await self._get_reference_images_for_command(event)

        # 发送确认
        msg = "已开始生图任务"
        if images_data:
            msg += f"[{len(images_data)}张参考图]"
        if matched_preset:
            msg += f"[预设: {matched_preset}]"

        logger.debug(
            f"[Gemini Image] 参数解析 - 消息: {msg}, 比例: {aspect_ratio}, 分辨率: {resolution}"
        )

        yield event.plain_result(msg)

        # 生成任务 ID
        task_id = hashlib.md5(f"{time.time()}{user_id}".encode()).hexdigest()[:8]

        # 记录任务摘要
        preset_name = matched_preset if matched_preset else "无"
        img_count = len(images_data) if images_data else 0
        logger.info(
            f"[Gemini Image] 任务摘要 [{task_id}] - 提示词: {prompt} | 预设: {preset_name} | 参考图: {img_count}张 | 分辨率: {resolution} | 比例: {aspect_ratio}"
        )

        # 创建后台任务
        self.create_background_task(
            self._generate_and_send_image_async(
                prompt=prompt,
                images_data=images_data or None,
                unified_msg_origin=event.unified_msg_origin,
                aspect_ratio=aspect_ratio,
                resolution=resolution,
                task_id=task_id,
            )
        )

    async def _fetch_images_from_event(
        self, event: AstrMessageEvent
    ) -> list[tuple[bytes, str]]:
        """从事件中提取所有相关图片（当前消息、引用消息、At用户头像）"""
        images_data = []

        if not event.message_obj.message:
            return images_data

        logger.debug(
            f"[Gemini Image] Searching images from event components: {event.message_obj.message}"
        )

        # 0. 预扫描：获取回复发送者ID和统计At次数
        reply_sender_id = None
        at_counts = {}

        for component in event.message_obj.message:
            if isinstance(component, Comp.Reply):
                if hasattr(component, "sender_id") and component.sender_id:
                    reply_sender_id = str(component.sender_id)
            elif isinstance(component, Comp.At):
                if component.qq != "all":
                    uid = str(component.qq)
                    at_counts[uid] = at_counts.get(uid, 0) + 1

        # 遍历消息组件
        for component in event.message_obj.message:
            # 1. 处理直接发送的图片
            if isinstance(component, Comp.Image):
                url = component.url or component.file
                if url and (data := await self._download_image(url)):
                    images_data.append(data)

            # 2. 处理引用消息中的图片
            elif isinstance(component, Comp.Reply):
                if component.chain:
                    for sub_comp in component.chain:
                        if isinstance(sub_comp, Comp.Image):
                            url = sub_comp.url or sub_comp.file
                            if url and (data := await self._download_image(url)):
                                images_data.append(data)

            # 3. 处理 At 用户（获取头像）
            elif isinstance(component, Comp.At):
                if component.qq != "all":  # 忽略 @全体成员
                    uid = str(component.qq)

                    # 核心逻辑：判断是否是引用消息带来的自动 @
                    if reply_sender_id and uid == reply_sender_id:
                        # 如果该 ID 只出现了一次 At，且是引用消息的发送者，则认为是自动 @，忽略头像
                        if at_counts.get(uid, 0) == 1:
                            logger.debug(
                                f"[Gemini Image] Ignoring auto-At for reply sender {uid}"
                            )
                            continue
                        # 如果出现多次，说明用户显式 @ 了（除了自动 @ 之外），保留

                    # 核心逻辑2：判断是否是触发机器人的 At
                    # 如果 Bot 被 At 了正好一次，通常是作为指令触发前缀，忽略头像
                    # 如果 Bot 被 At 了多次，说明用户显式引用了 Bot 头像
                    self_id = str(event.get_self_id()).strip()
                    if self_id and uid == self_id:
                        if at_counts.get(uid, 0) == 1:
                            logger.debug(
                                f"[Gemini Image] Ignoring bot trigger At {uid}"
                            )
                            continue

                    if avatar_data := await self.get_avatar(uid):
                        images_data.append((avatar_data, "image/jpeg"))

        return images_data

    async def _get_reference_images_for_command(
        self, event: AstrMessageEvent
    ) -> list[tuple[bytes, str]]:
        """为指令获取参考图片"""
        return await self._fetch_images_from_event(event)

    @filter.command("生图模型")
    async def model_command(self, event: AstrMessageEvent, model_index: str = ""):
        """生图模型管理指令"""
        user_id = str(event.get_sender_id() or event.unified_msg_origin)
        group_id = event.message_obj.group_id or ""

        if not self._check_permission(user_id, group_id):
            yield event.plain_result("❌ 您没有权限使用此功能")
            return

        if not model_index:
            model_list = ["📋 可用模型列表:"]
            for idx, model in enumerate(self.AVAILABLE_MODELS, 1):
                marker = " ✓" if model == self.model else ""
                model_list.append(f"{idx}. {model}{marker}")

            model_list.append(f"\n当前使用: {self.model}")
            yield event.plain_result("\n".join(model_list))
            return

        try:
            index = int(model_index) - 1
            if 0 <= index < len(self.AVAILABLE_MODELS):
                new_model = self.AVAILABLE_MODELS[index]
                self.model = new_model
                self.generator.model = new_model
                # 保存到分组配置
                if "api_config" not in self.config:
                    self.config["api_config"] = {}
                self.config["api_config"]["model"] = new_model
                self.config.save_config()
                yield event.plain_result(f"✅ 模型已切换: {new_model}")
            else:
                yield event.plain_result("❌ 无效的序号")
        except ValueError:
            yield event.plain_result("❌ 请输入有效的数字序号")

    @filter.command("预设")
    async def preset_command(self, event: AstrMessageEvent):
        """预设管理指令"""
        user_id = str(event.get_sender_id() or event.unified_msg_origin)
        group_id = event.message_obj.group_id or ""

        if not self._check_permission(user_id, group_id):
            yield event.plain_result("❌ 您没有权限使用此功能")
            return

        # user_id 已经是正确的ID了，不需要重新赋值（注意下面还有一行 event.unified_msg_origin 获取 mask_uid 的逻辑，可能需要调整）
        masked_uid = (
            user_id[:4] + "****" + user_id[-4:] if len(user_id) > 8 else user_id
        )

        message_str = (event.message_str or "").strip()
        logger.info(
            f"[Gemini Image] 收到预设指令 - 用户: {masked_uid}, 内容: {message_str}"
        )

        parts = message_str.split(maxsplit=1)

        cmd_text = ""
        if len(parts) > 1:
            cmd_text = parts[1].strip()

        if not cmd_text:
            if not self.presets:
                yield event.plain_result("📋 当前没有预设")
                return

            preset_list = ["📋 预设列表:"]
            for idx, (name, prompt) in enumerate(self.presets.items(), 1):
                display = prompt[:20] + "..." if len(prompt) > 20 else prompt
                preset_list.append(f"{idx}. {name}: {display}")
            yield event.plain_result("\n".join(preset_list))
            return

        if cmd_text.startswith("添加 "):
            parts = cmd_text[3:].split(":", 1)
            if len(parts) == 2:
                name, prompt = parts
                self.presets[name.strip()] = prompt.strip()
                # 保存
                self.config["presets"] = [f"{k}:{v}" for k, v in self.presets.items()]
                self.config.save_config()
                yield event.plain_result(f"✅ 预设已添加: {name.strip()}")
            else:
                yield event.plain_result("❌ 格式错误: /预设 添加 名称:内容")

        elif cmd_text.startswith("删除 "):
            name = cmd_text[3:].strip()
            if name in self.presets:
                del self.presets[name]
                self.config["presets"] = [f"{k}:{v}" for k, v in self.presets.items()]
                self.config.save_config()
                yield event.plain_result(f"✅ 预设已删除: {name}")
            else:
                yield event.plain_result(f"❌ 预设不存在: {name}")

    async def _get_reference_images_for_tool(
        self, event: AstrMessageEvent
    ) -> list[tuple[bytes, str]]:
        """获取参考图片列表（用于工具调用）"""
        # 从事件中获取（包含当前图片、引用图片、At头像）
        images_data = await self._fetch_images_from_event(event)
        return images_data

    def create_background_task(self, coro: Coroutine[Any, Any, Any]) -> asyncio.Task:
        """统一创建后台任务并追踪生命周期"""
        task = asyncio.create_task(coro)
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)
        return task

    @staticmethod
    async def get_avatar(user_id: str) -> bytes | None:
        """下载QQ用户头像"""
        url = f"https://q4.qlogo.cn/headimg_dl?dst_uin={user_id}&spec=640"
        try:
            # 复用 astrbot 的下载工具
            path = await download_image_by_url(url)
            if path:
                with open(path, "rb") as f:
                    return f.read()
        except Exception:
            pass
        return None

    async def _download_image(self, url: str) -> tuple[bytes, str] | None:
        """下载图片并返回数据与 MIME 类型 (Helper wrapper around core utility)"""
        try:
            data = None
            # 尝试作为本地文件读取
            if os.path.exists(url) and os.path.isfile(url):
                with open(url, "rb") as f:
                    data = f.read()
            else:
                path = await download_image_by_url(url)
                if path:
                    with open(path, "rb") as f:
                        data = f.read()

            if not data:
                return None

            # 检查大小
            if len(data) > self.max_image_size_mb * 1024 * 1024:
                logger.warning(
                    f"[Gemini Image] 图片超过大小限制 ({self.max_image_size_mb}MB)"
                )
                return None

            # 简单推断 mime
            mime = "image/png"
            if data.startswith(b"\xff\xd8"):
                mime = "image/jpeg"
            elif data.startswith(b"GIF"):
                mime = "image/gif"
            elif data.startswith(b"RIFF") and b"WEBP" in data[:16]:
                mime = "image/webp"

            return data, mime
        except Exception as e:
            logger.error(f"[Gemini Image] 获取图片失败 (URL/Path: {url}): {e}")
        return None

    async def _generate_and_send_image_async(
        self,
        prompt: str,
        unified_msg_origin: str,
        images_data: list[tuple[bytes, str]] | None = None,
        aspect_ratio: str = "1:1",
        resolution: str = "1K",
        task_id: str | None = None,
    ):
        """异步生成图片并发送"""
        if not task_id:
            task_id = hashlib.md5(
                f"{time.time()}{unified_msg_origin}".encode()
            ).hexdigest()[:8]

        # 处理 "自动" 比例
        final_ar = aspect_ratio
        if aspect_ratio == "自动":
            final_ar = None

        async with self._generation_semaphore:
            try:
                results, error = await self.generator.generate_image(
                    prompt=prompt,
                    images_data=images_data,
                    aspect_ratio=final_ar,
                    image_size=resolution,
                    task_id=task_id,
                )

                if error:
                    await self.context.send_message(
                        unified_msg_origin,
                        MessageChain().message(f"❌ 生成失败: {error}"),
                    )
                    return

                if not results:
                    return

                logger.info(
                    f"[Gemini Image] 任务完成 [{task_id}] - 生成了 {len(results)} 张图片"
                )

                # 构建消息链
                chain = MessageChain()
                cached_urls = []

                for img_bytes in results:
                    # 保存临时文件
                    try:
                        file_path = save_temp_img(img_bytes)
                        chain.file_image(file_path)
                        cached_urls.append(f"file://{file_path}")
                    except Exception as e:
                        logger.error(f"保存图片失败: {e}")

                await self.context.send_message(unified_msg_origin, chain)

            except Exception as e:
                logger.error(f"[Gemini Image] 任务失败: {e}", exc_info=True)
                await self.context.send_message(
                    unified_msg_origin,
                    MessageChain().message("❌ 生成过程中发生未知错误"),
                )

    async def terminate(self):
        """卸载清理"""
        try:
            # 1. 关闭生成器 session
            if self.generator:
                await self.generator.close_session()

            # 2. 取消后台任务
            for task in list(self.background_tasks):
                if not task.done():
                    task.cancel()

            logger.info("[Gemini Image] 插件已卸载")

        except Exception as e:
            logger.error(f"[Gemini Image] 卸载清理出错: {e}")
