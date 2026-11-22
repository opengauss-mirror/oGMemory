"""Delivery submission and receipt completion."""

from datetime import UTC, datetime

from memorybot.models import BotError


def run(execution):
    delivery = execution.row(
        "bot_delivery", {"delivery_id": execution.task["input"]["delivery_id"]}
    )
    if delivery["channel"] != "web":
        raise BotError("DELIVERY_UNAVAILABLE", "交付目标必须为 Web 会话", 409)
    receipt = {
        "provider": "web",
        "session_id": delivery["target_id"],
        "source_task_id": delivery["source_task_id"],
        "message_id": delivery["delivery_id"],
        "confirmed_at": datetime.now(UTC).isoformat(),
    }

    def complete(tx):
        if tx.task["cancel_requested"]:
            return
        tx.update(
            "bot_delivery",
            {"delivery_id": delivery["delivery_id"]},
            {"state": "sent", "provider_receipt": receipt},
        )
        execution.store.append_event(
            tx,
            delivery["source_task_id"],
            "delivery.updated",
            {
                "delivery_id": delivery["delivery_id"],
                "state": "sent",
                "channel": delivery["channel"],
                "target_id": delivery["target_id"],
            },
        )

    # Commit the receipt and source-task event with the delivery task's completion.
    execution.store.finish(*execution.identity, delivery["content"], mutate=complete)
