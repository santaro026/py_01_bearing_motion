"""
Created on Wed Apr 08 18:17:03 2026
@author: santaro



"""

import numpy as np
import polars as pl
import matplotlib.pyplot as plt

from pathlib import Path
import datetime

today = datetime.datetime.now()

def extract_list():
    df_iam = pl.read_excel(datadir/"members_list.xlsx", sheet_name="iam", drop_empty_cols=True, drop_empty_rows=True)
    df_iam = (df_iam
            .with_row_index("pid")
            .with_columns(
                group=pl.col("departmentName").str.slice(7).str.strip_chars(),
                full_name=normalize_kanji(pl.col("name").str.strip_chars()),
                short_name=normalize_kanji(pl.col("name").str.split(" ").list.first().str.strip_chars())
            )
            .select(pl.col("pid", "group", "full_name", "short_name"))
            #   .filter(pl.col("group").is_in(["DTO1", "DTO2", "DTO3", "DTO4"]))
            )
    # print(df_iam)
    df_iam.write_csv(outdir/"iam.csv")
    df_iam.write_json(outdir/"iam.json")

    #### attendees list
    df_attendee = pl.read_excel(datadir/"members_list.xlsx", sheet_name="attendee", drop_empty_cols=True, drop_empty_rows=True)
    df_attendeel = (
        df_attendee.with_row_index("row_id")
        .unpivot(index="row_id", variable_name="group", value_name="short_name")
        .drop_nulls()
        .with_columns(
            group=pl.col("group"),
            short_name=normalize_kanji(pl.col("short_name").str.strip_chars())
        )
        .join(
            df_iam,
            on=["group", "short_name"],
            how="left"
        )
        .with_columns(
            group=pl.col("group").str.strip_chars(),
            short_name=normalize_kanji(pl.col("short_name").str.strip_chars()),
            full_name=normalize_kanji(pl.col("full_name").str.strip_chars())
        )
        .select(pl.col("pid", "group", "full_name", "short_name"))
    )
    # print(df_attendeel)
    df_attendeel.write_csv(outdir/"attendeel.csv")
    df_attendeel.write_json(outdir/"attendeel.json")

    #### check number of all attendees
    # count = df_attendeel.group_by(pl.col("group")).agg(pl.len().alias("appearance_count"))
    # print(count)

    df_presenter = pl.read_excel(datadir/"members_list.xlsx", sheet_name="presenter", drop_empty_cols=True, drop_empty_rows=True)
    df_presenter = (
        df_presenter
        .with_row_index("row_id")
        .with_columns(
            group=pl.col("group").str.strip_chars(),
            short_name=normalize_kanji(pl.col("presenter").str.strip_chars()).str.split("・"),
        )
        .select(pl.col("row_id", "group", "short_name"))
        .explode("short_name")
    )
    # print(df_presenter)

    df_presenterl = (
        df_presenter
        .join(
            df_iam,
            on=["group", "short_name"],
            how="left"
        )
        .select(pl.col("row_id", "pid", "group", "full_name", "short_name"))
        .group_by("row_id", maintain_order=True)
        .agg(
            pl.col("pid", "group", "full_name", "short_name"),
        )
    )
    # print(df_presenterl)
    df_presenterlc = (
        df_presenterl.with_columns(
            pid=pl.col("pid").list.eval(pl.element().cast(pl.String)).list.join(";"),
            group=pl.col("group").list.join(";"),
            full_name=pl.col("full_name").list.join(";"),
            short_name=pl.col("short_name").list.join(";"),
        )
    )
    # print(df_presenterlc)
    df_presenterlc.write_csv(outdir/"presenterl.csv")
    df_presenterl.write_json(outdir/"presenterl.json")

    #### check duplication
    # df_presenter = (
        # .filter(df_presenter.is_duplicated())
        # df_presenter.filter(~pl.col("presenter").is_in("raw")).select("presenter")
    # )
    # print(df_presenter)
    # presenter = df_presenter.get_column("presenter").drop_nulls().unique().sort()
    # raw = df_presenter.get_column("raw").drop_nulls().unique().sort()
    # print(f"same: {presenter.equals(raw)}")

def aggregate_record():
    df_materials = (
        pl.read_csv(datadir/"tasktalk_materials.csv")
        .with_columns(pl.col("発表日").str.strptime(pl.Date, "%Y/%m/%d").alias("date"))
        .select("date", "Title", "発表者", "グループ")
    )
    # print(df_materials.columns)
    # print(df_materials)

    df_materials_recent = df_materials.filter(
        pl.col("date") > pl.date(2025, 9, 1)
        # pl.col("date") > pl.date(2025, 1, 1)
        # pl.col("date") > pl.date(2020, 1, 1)
    )
    # print(df_materials_recent)

    # count = df_materials_recent.group_by("発表者").agg(pl.len().alias("appearance_count")).sort("appearance_count", descending=True)
    # print(count)

    # count.write_json("count.json")
    # count.write_csv("count_all.csv")

def get_weekday(weekday, weeks=0, year=today.year, month=today.month):
    first_day = datetime.date(year, month, 1)
    wd = first_day.weekday()
    days_to_add = weekday - wd
    if days_to_add < 0: days_to_add = days_to_add + 7
    weeks_to_add = weeks
    target_day = first_day + datetime.timedelta(days=days_to_add+7*weeks_to_add)
    return target_day


def get_holidays(today=today):
    holidays = {
        "New Year's Day": datetime.date(today.year, 1, 1),
        "Coming-of-Age Day": get_weekday(1, weeks=1, month=1),
        "National Foundation Commemoration Day": datetime.date(today.year, 2, 11),
        "Emperor's Birthday": datetime.date(today.year, 2, 23),
        "Vernal Equinox Day": datetime.date(today.year, 3, 20),
        "Showa Day": datetime.date(today.year, 4, 29),
        "Constitution Memorial Day": datetime.date(today.year, 5, 3),
        "Greenery Day": datetime.date(today.year, 5, 4),
        "Greenery Day": datetime.date(today.year, 5, 5),
        "Marine Day": get_weekday(0, weeks=2, month=6),
        "Mountain Day": datetime.date(today.year, 8, 11),
        "Respect for the Aged Day": get_weekday(0, weeks=2, month=9),
        "Autumnal Equinox Day": datetime.date(today.year, 9, 23),
        "Sports Day": get_weekday(0, weeks=1, month=10),
        "Culture Day": datetime.date(today.year, 11, 3),
        "Labor Thanksgiving Day": datetime.date(today.year, 11, 23),
    }
    return holidays


def get_myholidays(today=today):
    holidays_firsthalf = [
        [datetime.date(2026, 4, 25), datetime.date(2026, 5, 6)],
        [datetime.date(2026, 7, 20), datetime.date(2026, 7, 20)],
        [datetime.date(2026, 8, 7), datetime.date(2026, 8, 18)],
        [datetime.date(2026, 9, 19), datetime.date(2026, 9, 22)],
    ]
    holidays = []
    for _drange in holidays_firsthalf:
        _d = _drange[0]
        while _d <= _drange[1]:
            holidays.append(_d)
            _d = _d + datetime.timedelta(days=1)
    # print(holidays)

if __name__ == "__main__":
    print("---- run ----")

    start_date = datetime.date(2026, 4, 16)
    step = datetime.timedelta(weeks=2)
    dates = []
    cur = start_date
    while cur < datetime.date(2026, 10, 1):
        dates.append(cur)
        cur = cur + step
    # print(dates)

    import pprint
    a = get_holidays()
    pprint.pprint(a)

    # get_holidays()


    # df = pl.DataFrame(holidays)
    # df.write_json(outdir/"test.json")








