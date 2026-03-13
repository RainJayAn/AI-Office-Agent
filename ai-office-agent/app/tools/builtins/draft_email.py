from typing import Any


def draft_email(
    recipient: str,
    subject: str,
    purpose: str,
    key_points: list[str] | None = None,
) -> str:
    points = key_points or []
    points_block = ""
    if points:
        rendered_points = "\n".join(f"- {item}" for item in points)
        points_block = f"\n\n关键要点：\n{rendered_points}"

    return (
        f"邮件主题：{subject}\n\n"
        f"收件人：{recipient}\n\n"
        "您好，\n\n"
        f"此次来信主要是想说明：{purpose}。"
        f"{points_block}\n\n"
        "如需我补充更多信息，我会继续跟进。\n\n"
        "此致\n"
        "敬礼"
    )


def build_draft_email_tool() -> dict[str, Any]:
    return {
        "name": "draft_email",
        "description": "Draft a professional email from structured fields.",
        "func": draft_email,
    }
