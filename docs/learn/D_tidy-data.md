# 付録 D. 解析のためのデータの準備

解析の多くは、表を一つ読み込むだけでは終わりません。所見の表に検査の情報をつける、検査の表に所見の数をつける、患者ごとに検査を時間の順に並べる、といった作業が要ります。こうした**表の結合**や**時系列の集計**にあとからつなげるには、CSV（comma-separated values）ファイルを編集するときに、**整然データ**（tidy data）の形を保つことが大切です（Wickham 2014）。この付録では、内視鏡の検査一覧と所見一覧を例に、準備のときに守ることをまとめます。

[← 目次](index.md) · [← 付録 C](C_p-value.md)

## D.1 整然データとは {#tidy}

整然データは、次の三つを満たす表です（Wickham 2014）。

- 1 行が 1 つの観測（1 検査、1 所見、1 患者など）
- 1 列が 1 つの変数
- 1 セルが 1 つの値

Excel の台帳は、見やすさのために、この形を崩しがちです。たとえば次の表は、人が読むには困りませんが、解析ソフトはそのままでは扱えません。

| 検査 | 日付 | 病変 | 大きさ |
|------|------|------|--------|
| 1 | 2024/4/1 | 直腸 0-IIa、S 状結腸 0-Is | 15mm、8mm |
| 2 | 4月3日 | 盲腸 LST-G | 30 mm |

- 1 セルに病変が二つ（「直腸 0-IIa、S 状結腸 0-Is」）
- 部位と肉眼型が同じセル
- 単位つきの数（「15mm」）
- 日付の書き方の混在

整然データにすると、所見一覧は 1 行 1 病変になり、検査の情報は検査一覧に分かれます。

| id_item | id_proc | 部位 | 肉眼型 | 大きさ_mm |
|---------|---------|------|--------|-----------|
| 101 | 1 | 直腸 | 0-IIa | 15 |
| 102 | 1 | S 状結腸 | 0-Is | 8 |
| 103 | 2 | 盲腸 | LST-G | 30 |

| id_proc | 日付 |
|---------|------|
| 1 | 2024-04-01 |
| 2 | 2024-04-03 |

日付は 2024-04-01 の形（ISO 8601）にそろえます。列名は、実際には英数字にするほうが安全です（[D.3](#checklist)）。ここでは読みやすさのために日本語にしました。

## D.2 表とキー {#keys}

検査一覧と所見一覧は、**キー**でつながります。キーは、行を一つに決める列です。

| 表 | 1 行の単位 | プライマリーキー | ほかの表を指す列 |
|----|-----------|------------------|------------------|
| 検査一覧 | 1 検査 | dump `id_proc` / DuckDB `hid_proc` | 患者 ID |
| 所見一覧 | 1 所見 | dump `id_item` / DuckDB `hid_item` | `id_proc` / `hid_proc` |

**プライマリーキー**（primary key）は、表の中で値が重ならない列です。所見一覧の `id_proc` は、その所見がどの検査のものかを示します。1 回の検査に所見が複数あれば、所見一覧では同じ `id_proc` が何行にも出てきます。この「1 つの検査に複数の所見」という関係を使って、二つの表を結合します。

キーの値が一つでも書き換わると、その行はほかの検査の行と結合されます。エラーは出ないので、気づかないまま誤った結果になります。

## D.3 準備のとき守ること {#checklist}

CSV を Excel などで編集するときは、次の点を守ります。

1. **識別列は削除、書き換え、複製をしません。** dump の `id_proc` と `id_item`、DuckDB の `hid_proc` と `hid_item` です。壊すと結合がずれます。
2. **各表のプライマリーキーを 1 つ残します。** 検査一覧なら dump の `id_proc`（DuckDB の `hid_proc`）、所見一覧なら `id_item`（`hid_item`）です。
3. **1 セルには 1 つの値を入れます。** カンマで複数の値を並べず、行を分けるか、別の列にします。
4. **決まったカテゴリの列（肉眼型、病理、部位など）に、一覧にない値を入れません。** 「直腸」と「Rb」と「直腸 」（末尾に空白）は、集計で別の項目に分かれます。
5. **列の追加、行の削除、並べ替えはしてかまいません。** 行を削除するときは、何行を削除したかを記録します。行を残したまま `exclude` 列に印をつけ、解析のときに除く方法もあります（[D.6](#exclude)）。
6. **列名は重ならないようにします。** スペース、改行、括弧は避けます。
7. **1 行目は列名、2 行目からはデータにします。** 列名の行が 2 行あると、読み込みがずれます。

Excel では、次のことも起きやすいので注意します。

- セルの結合
- 色や太字で意味をつけること（CSV に保存すると消えます）
- 欠測を 0 や「-」で書くこと（空欄と区別がつかなくなります）
- 数字の ID が日付や指数表記に変わること（`1-2` が 1 月 2 日に、長い数字が `1.23E+11` に）

ほかに気をつけたい点は、Broman と Woo（2018）の論文にまとまっています。

## D.4 表を結合する {#join}

ここからは、整った表を Python の polars で読み込んで使う例です。statract は polars のデータフレームを受け取ります。

まず、キーが本当に一意かを確かめます。

```python
import polars as pl

proc = pl.read_csv("procedures.csv", try_parse_dates=True)  # 検査一覧：1 行 1 検査
item = pl.read_csv("items.csv")                             # 所見一覧：1 行 1 所見

assert proc["id_proc"].is_unique().all()
assert item["id_item"].is_unique().all()
```

**所見一覧に検査の情報をつける**例です。所見ごとに、その検査でミダゾラムを使ったかどうかがわかります。

```python
item_with_proc = item.join(
    proc.select("id_proc", "midazolam"),
    on="id_proc",
    how="left",
    validate="m:1",   # 所見は多、検査は 1。検査一覧に同じ id_proc が 2 行あればエラー
)
```

**検査一覧に所見の情報をつける**例です。検査ごとに、浸潤癌の所見がいくつあったかを数えます。

```python
n_invasive = item.group_by("id_proc").agg(
    (pl.col("pathology") == "invasive").sum().alias("n_invasive")
)
proc_with_item = proc.join(n_invasive, on="id_proc", how="left", validate="1:1").with_columns(
    pl.col("n_invasive").fill_null(0)   # 所見のない検査は 0 個
)
```

`validate` を指定すると、キーが崩れていたときに結合の時点でエラーになります。結合の前後で行数が変わっていないかも確かめます。

## D.5 患者ごとの時系列 {#timeline}

検査一覧に患者 ID（ここでは `id_patient`）と日付があれば、患者ごとに検査を時間の順に並べられます。例として、内視鏡的粘膜下層剥離術（endoscopic submucosal dissection、ESD）のあと 180 日以内に、狭窄に対する内視鏡的バルーン拡張術（endoscopic balloon dilation、EBD）を受けたかを、ESD ごとに数えます。

```python
esd = proc.filter(pl.col("procedure") == "ESD").select("id_proc", "id_patient", "date")
ebd = proc.filter(pl.col("procedure") == "EBD").select(
    "id_patient", pl.col("date").alias("date_ebd")
)

esd_ebd = (
    esd.join(ebd, on="id_patient", how="left")
    .with_columns(
        ((pl.col("date_ebd") > pl.col("date"))
         & (pl.col("date_ebd") - pl.col("date") <= pl.duration(days=180))).alias("ebd_180")
    )
    .group_by("id_proc")
    .agg(pl.col("ebd_180").any())   # EBD のない ESD は false
)
```

この集計は、`procedure` 列の値が「ESD」と「EBD」にそろっていて、日付が日付として読み込めることを前提にしています。「ESD 」や「esd」が混ざると、その検査は数えられません。ここでも [D.3](#checklist) の 4 番目が効いてきます。

追跡期間と打ち切りを扱う解析（生存時間解析）に進むときは、同じように患者ごとの時系列から、追跡の長さ（`time`）とイベントの有無（`event`）の列を作ります。考え方は [第 16 章](16_survival.md) にあります。

## D.6 行を除くとき {#exclude}

行を削除すると、何人を、なぜ除いたのかが残りません。論文では、対象の選び方と除外の数を示す必要があります（[付録 B](B_reporting-guidelines.md) の STROBE など）。そこで、行は残したまま、`exclude` 列に除外の理由を書いておく方法があります。

| id_proc | … | exclude |
|---------|---|---------|
| 1 | … | |
| 2 | … | 同意なし |
| 3 | … | 他院で治療済み |

```python
print(proc["exclude"].value_counts())        # 除外の理由ごとの数
analysis = proc.filter(pl.col("exclude").is_null())
```

除外の理由ごとの数が、そのまま対象者の流れ図（フローチャート）の数字になります。

## D.7 読み込んだあとの確認 {#check}

解析の前に、読み込んだ表が D.3 を満たしているかを確かめます。

```python
# 列名の重なり。polars は重なった列名に _duplicated_ をつけて読み込みます
assert not any("_duplicated_" in c for c in item.columns)

# カテゴリの列に一覧にない値がないか
allowed = ["直腸", "S 状結腸", "下行結腸", "横行結腸", "上行結腸", "盲腸"]
print(item.filter(~pl.col("site").is_in(allowed))["site"].unique())

# 数値の範囲と欠測
print(item.describe())
```

問題が見つかったら、元の CSV を直すか、直す手順をスクリプトに書きます。どちらにしても、何をどう直したかが、あとからわかるようにしておきます。表が整えば、[第 1 章](01_descriptive.md#table1) の Table 1 から解析を始められます。

## 文献 {#references}

- Wickham H. Tidy data. Journal of Statistical Software 2014;59(10):1–23.
- Broman KW, Woo KH. Data organization in spreadsheets. The American Statistician 2018;72(1):2–10.

## まとめ

- 整然データは、1 行 1 観測、1 列 1 変数、1 セル 1 値の表です。
- 検査一覧と所見一覧は、プライマリーキー（`id_proc`、`id_item`）でつながります。識別列は削除、書き換え、複製をしません。
- カテゴリの列には一覧にある値だけを入れます。表記ゆれは集計で別の項目になります。
- 行を除くときは数を記録するか、`exclude` 列に理由を書いて、解析のときに除きます。
- 結合では `validate` を指定し、前後の行数を確かめます。

**最後のページです。**[目次に戻る](index.md)
