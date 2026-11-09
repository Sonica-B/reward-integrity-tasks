NUMERIC = "numeric"
CATEGORY = "category"


def candidate_splits(kind, values):
    full = (1 << len(values)) - 1
    if kind == NUMERIC:
        points = sorted(set(values))
        cuts = [(lo + hi) / 2 for lo, hi in zip(points, points[1:])]
        splits = [(f"> {cut:g}", sum(1 << i for i, v in enumerate(values) if v > cut)) for cut in cuts]
    elif kind == CATEGORY:
        splits = [(f"== {value!r}", sum(1 << i for i, v in enumerate(values) if v == value))
                  for value in sorted(set(values), key=repr)]
    else:
        raise ValueError(f"unknown feature kind {kind!r}")
    return [(text, mask) for text, mask in splits if 0 < mask < full]


def split_table(table):
    return {name: candidate_splits(kind, values) for name, (kind, values) in table.items()}


def leaf(region, target):
    hit = target & region
    if hit == region:
        return "legit"
    if hit == 0:
        return "exploit"
    return None


def stump(mask, region, target):
    inside, want = mask & region, target & region
    if inside == want:
        return "legit"
    if inside == region ^ want:
        return "exploit"
    return None


def side_rules(splits, region, target):
    pure = leaf(region, target)
    if pure is not None:
        return pure, {}
    found = {}
    for name, options in splits.items():
        for text, mask in options:
            label = stump(mask, region, target)
            if label is not None:
                found.setdefault(name, f"({name} {text} -> {label} else "
                                       f"{'exploit' if label == 'legit' else 'legit'})")
                break
    return None, found


def depth1_rules(table, labels):
    full = (1 << len(labels)) - 1
    target = sum(1 << i for i, kind in enumerate(labels) if kind == "legit")
    rules = []
    for name, options in split_table(table).items():
        for text, mask in options:
            label = stump(mask, full, target)
            if label is not None:
                rules.append(f"{name} {text} -> {label}")
    return sorted(rules)


def depth2_rules(table, labels, pairs_only=True):
    full = (1 << len(labels)) - 1
    target = sum(1 << i for i, kind in enumerate(labels) if kind == "legit")
    splits = split_table(table)
    rules = []
    for root, options in splits.items():
        for text, mask in options:
            pure_in, rules_in = side_rules(splits, mask, target)
            pure_out, rules_out = side_rules(splits, full ^ mask, target)
            if pure_in is None and not rules_in or pure_out is None and not rules_out:
                continue
            for name in splits:
                use_in = [pure_in] if pure_in else [rules_in[n] for n in {root, name} if n in rules_in]
                use_out = [pure_out] if pure_out else [rules_out[n] for n in {root, name} if n in rules_out]
                if not pairs_only:
                    use_in = [pure_in] if pure_in else list(rules_in.values())[:1]
                    use_out = [pure_out] if pure_out else list(rules_out.values())[:1]
                if use_in and use_out:
                    rules.append(f"if {root} {text}: {use_in[0]} else: {use_out[0]}")
                    if not pairs_only:
                        break
    return sorted(set(rules))
