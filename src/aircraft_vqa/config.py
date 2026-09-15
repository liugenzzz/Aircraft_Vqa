"""数据源配置的加载与合并。

configs/datasets.yaml 是**入库的模板**，记录每个源怎么读。
哪些源在本机已经下好、该不该启用，是每台机器自己的事，写在
configs/datasets.local.yaml（不入库）里，形如：

    datasets:
      - name: aircraft_skin_defects
        enabled: true

本地这份只需要写要改的字段，其余从模板继承 —— 所以脚本不用再去改
入库文件，`git pull` 也就不会再和本地状态打架。
"""
from __future__ import annotations

import os
from typing import Optional

import yaml

LOCAL_SUFFIX = ".local.yaml"


def local_path_for(config_path: str) -> str:
    """configs/datasets.yaml -> configs/datasets.local.yaml"""
    base = config_path[:-5] if config_path.endswith(".yaml") else config_path
    return base + LOCAL_SUFFIX


def merge_dataset_specs(groups: list) -> list:
    """按 name 浅合并多份配置，后来者覆盖前者的**同名字段**。

    注意是浅合并而不是整条替换：本地那份只写 `{name, enabled}`，
    整条替换会把 adapter / root / class_map 全抹掉。
    """
    order, by_name = [], {}
    for specs in groups:
        for sp in specs or []:
            n = sp.get("name")
            if not n:
                continue
            if n not in by_name:
                by_name[n] = dict(sp)
                order.append(n)
            else:
                by_name[n].update(sp)
    return [by_name[n] for n in order]


def load_dataset_configs(paths: Optional[list] = None,
                         use_local: bool = True,
                         warn=print) -> tuple:
    """读配置，返回 (datasets, default_data_root)。

    每个显式给出的配置，都会顺带找一下它旁边的 .local.yaml 叠上去。
    """
    paths = paths or ["configs/datasets.yaml"]
    groups, default_root = [], "~/data/raw"
    for cp in paths:
        if not os.path.exists(cp):
            if warn:
                warn(f"[warn] 配置不存在，跳过：{cp}")
            continue
        part = yaml.safe_load(open(cp, encoding="utf-8")) or {}
        groups.append(part.get("datasets") or [])
        default_root = part.get("defaults", {}).get("data_root", default_root)

        if not use_local:
            continue
        lp = local_path_for(cp)
        if os.path.exists(lp):
            loc = yaml.safe_load(open(lp, encoding="utf-8")) or {}
            groups.append(loc.get("datasets") or [])
            default_root = loc.get("defaults", {}).get("data_root", default_root)
    return merge_dataset_specs(groups), default_root


def set_enabled(config_path: str, names: list, value: bool = True) -> list:
    """在**本地那份**里把这些源标成 enabled，不碰入库的模板。

    返回实际发生变化的 name。
    """
    if not names:
        return []
    lp = local_path_for(config_path)
    doc = {}
    if os.path.exists(lp):
        doc = yaml.safe_load(open(lp, encoding="utf-8")) or {}
    specs = doc.get("datasets") or []
    by_name = {s["name"]: s for s in specs if s.get("name")}

    changed = []
    for n in names:
        cur = by_name.get(n)
        if cur is None:
            cur = {"name": n}
            specs.append(cur)
            by_name[n] = cur
        if cur.get("enabled") != value:
            cur["enabled"] = value
            changed.append(n)
    if not changed:
        return []

    doc["datasets"] = specs
    os.makedirs(os.path.dirname(lp) or ".", exist_ok=True)
    with open(lp, "w", encoding="utf-8") as f:
        f.write("# 本机状态，不入库。哪些源已经下好、该不该启用写在这里。\n"
                "# 只写要改的字段，其余从 datasets.yaml 继承。\n")
        yaml.safe_dump(doc, f, allow_unicode=True, sort_keys=False)
    return changed
