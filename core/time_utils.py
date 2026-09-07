from __future__ import annotations

from datetime import datetime, timedelta, timezone


STANDARD_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
SHANGHAI_TZ = timezone(timedelta(hours=8))


def format_timestamp(value: str | datetime | None = None) -> str:
    """将时间统一转换为上海时区的标准字符串格式。"""
    if value is None or (isinstance(value, str) and not value.strip()):
        moment = datetime.now(SHANGHAI_TZ)
    elif isinstance(value, datetime):
        moment = value
    else:
        text = str(value).strip()
        # Python 的 fromisoformat 不直接识别所有版本中的 Z 后缀。
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        try:
            moment = datetime.fromisoformat(text)
        except ValueError as error:
            raise ValueError(
                f"时间格式无效，应为 {STANDARD_TIME_FORMAT}: {value}"
            ) from error

    # 无时区时间按项目默认的上海时区解释；有时区时间先转换后再移除时区信息。
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=SHANGHAI_TZ)
    else:
        moment = moment.astimezone(SHANGHAI_TZ)
    return moment.strftime(STANDARD_TIME_FORMAT)


__all__ = ["SHANGHAI_TZ", "STANDARD_TIME_FORMAT", "format_timestamp"]
