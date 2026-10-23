"""Read cctl own-pool lists across CLI and API-envelope output versions."""


def own_pool_rows(response):
    value = response
    for _ in range(4):
        if isinstance(value,list):
            if any(not isinstance(row,dict) for row in value):
                raise ValueError("Invalid cctl own-pool row")
            return value
        if not isinstance(value,dict):
            break
        if value.get("success") is False or value.get("ok") is False:
            raise ValueError("cctl own-pool lookup reported failure")
        for key in ("pools","data"):
            if key in value:
                value = value[key]
                break
        else:
            break
    raise ValueError("Unrecognized cctl own-pool response; expected a list, pools or data envelope")


def authorized_pool(rows):
    return any(row.get("name")=="model-autosft" or row.get("alias")=="model-autosft" for row in rows)


def pool_summary(rows):
    # Deliberately expose only pool identifiers, never arbitrary API metadata.
    return ", ".join(str(row.get("name") or row.get("alias") or "unnamed")[:96]
                     for row in rows[:20]) or "(empty own-pool list)"
