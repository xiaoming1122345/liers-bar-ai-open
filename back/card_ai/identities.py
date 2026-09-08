from __future__ import annotations

from .types import PlayerIdentity


def build_default_identities(
    player_count: int,
    hero_seat: int | None = None,
    custom_ids: dict[int, str] | None = None,
    hero_label: str = "我",
) -> tuple[PlayerIdentity, ...]:
    custom_ids = custom_ids or {}
    identities = []
    for seat in range(1, player_count + 1):
        if hero_seat is not None and seat == hero_seat:
            player_id = custom_ids.get(seat, "me")
            display_name = hero_label
            identities.append(
                PlayerIdentity(
                    seat=seat,
                    player_id=player_id,
                    display_name=display_name,
                    is_hero=True,
                )
            )
            continue

        default_name = f"玩家{seat}"
        player_id = custom_ids.get(seat, f"player_{seat}")
        display_name = custom_ids.get(seat, default_name)
        identities.append(
            PlayerIdentity(
                seat=seat,
                player_id=player_id,
                display_name=display_name,
                is_hero=False,
            )
        )
    return tuple(identities)
