# 付録 D. 解析のためのデータの準備

表の結合や時系列の集計にあとからつなげるには、CSV（comma-separated values）を編集するときに**整然データ**（tidy data）の形を保つことが大切です（Wickham 2014）。この付録では、崩れた台帳を一つ整え（D.1）、守ることをまとめ（D.2、D.3）、整った表を DuckDB の SQL と Python の polars で扱う例を示します（D.4–D.7）。

[← 目次](index.md) · [← 付録 C](C_p-value.md)

## D.1 崩れた台帳を整える {#example}

<div class="tidy-link" markdown>

**だめな例**：大腸内視鏡の Excel 台帳

| 患者 | 検査 | | | 病変 | | | | 病理 |
|------|------|---|---|------|---|---|---|------|
| | 日付 | 種類 | 鎮静 | 1 | 大きさ | 2 | 大きさ | |
| 10 | 2024/4/1 | ESD | ミダゾラム | 直腸 0-IIa | 15mm | S状 0-Is | 8 | 癌（SM）、腺腫 |
| 〃 | 5月1日 | EBD | あり | - | | | | |
| 11 | R6.4.3 | CS | なし | 盲腸 LST-G | 30 mm | | | 腺腫 |
| 計 | | | | 3 | | 1 | | |

この台帳の崩れ方と、それで起きること、直し方です。多くはエラーが出ず、**誤った数字が出ます**。

| 崩れ方 | 台帳では | 起きること | 直し方 |
|--------|----------|------------|--------|
| 2 行の列名 | 「検査」の下に「日付」 | 2 行目がデータとして読まれる | 列名は 1 行目だけ、英数字で |
| 省略やセル結合 | 「〃」 | CSV では空欄になり、誰の検査かわからない | 毎行に `id_patient` |
| ID がない | 検査、病変に番号なし | 表を結合できない | `id_proc`、`id_item` を追加 |
| 日付の混在 | 2024/4/1、5月1日、R6.4.3 | 日付として読めず、間隔を計算できない | 2024-04-01 の形（ISO 8601） |
| 1 列に複数の変数 | 「直腸 0-IIa」 | 部位ごとに集計できない | `site` と `morphology` に分ける |
| 単位入りの数 | 15mm、30 mm | 文字として読まれ、平均を計算できない | `size_mm` に数字だけ |
| 横に並べた病変 | 病変 1、病変 2 | 病変 3 つ目で列が足りない。集計に全部の列を見る | 所見一覧に 1 行 1 病変 |
| 1 セルに複数の値 | 「癌（SM）、腺腫」 | どの病変の病理かわからない。数えられない | 病変ごとの `pathology` |
| 表記ゆれ | 「ミダゾラム」と「あり」 | 集計で別の項目に分かれる | 決まった値だけ（`midazolam` = 1 / 0） |
| 欠測の混在 | 「-」と空欄 | 病変なしか書き忘れか区別できない | 欠測は空欄。病変なしは行を作らない |
| 合計の行 | 「計」 | 1 件の観測として数えられる | 消して、解析のときに数える |
| 2 種類の観測が同じ行 | 検査と病変が 1 行 | 所見の多い検査ほど重く数えられる | 検査一覧と所見一覧に分ける |

</div>

**よい例**：同じデータを、検査一覧（1 行 1 検査）と所見一覧（1 行 1 所見）に分けます。

| id_proc | id_patient | date | procedure | midazolam |
|---------|------------|------|-----------|-----------|
| 1 | 10 | 2024-04-01 | ESD | 1 |
| 2 | 11 | 2024-04-03 | colonoscopy | 0 |
| 3 | 10 | 2024-05-01 | EBD | 1 |

| id_item | id_proc | site | morphology | size_mm | pathology |
|---------|---------|------|------------|---------|-----------|
| 101 | 1 | rectum | 0-IIa | 15 | invasive |
| 102 | 1 | sigmoid | 0-Is | 8 | adenoma |
| 103 | 2 | cecum | LST-G | 30 | adenoma |

崩れ方の表の最後の行（2 種類の観測が同じ行）は、台帳を所見ごとの行に直しても起きます。所見の行に検査の情報（ミダゾラムあり、あり、なし）を毎回書くと、ミダゾラムを使った検査の割合が 3 行中 2 行で 67% になります。本当は 2 検査中 1 検査で 50% です。

色や太字の意味づけ（「赤は癌」）も、CSV にすると消えます。意味は `malignant` = 1 のような列で表します。

ID は、編集の途中でも壊れます。フィルハンドルでコピーすると連番に変わり、`id_proc` の列だけを並べ替えると別の検査を指します。長い数字は `1.23E+11` のような指数表記に変わります。どれもエラーは出ず、別の検査の所見として結合されます。

## D.2 整然データの決まりとキー {#tidy}

整然データは、次の三つを満たす表です（Wickham 2014）。

- 1 行が 1 つの観測（ここでは 1 検査、1 所見）
- 1 列が 1 つの変数
- 1 セルが 1 つの値

表どうしは**キー**でつながります。**プライマリーキー**（primary key）は、表の中で値が重ならない列です。

| 表 | 1 行の単位 | プライマリーキー | ほかの表を指す列 |
|----|-----------|------------------|------------------|
| 検査一覧 | 1 検査 | `id_proc` | `id_patient`（患者） |
| 所見一覧 | 1 所見 | `id_item` | `id_proc`（検査） |

検査 1 には所見が二つあるので、所見一覧では `id_proc` = 1 が 2 行あります。この関係で二つの表を結合します（[D.5](#join)）。

## D.3 準備のとき守ること {#checklist}

1. **識別列（`id_proc`、`id_item`、`id_patient`）は削除、書き換え、複製をしません。** 壊すと結合がずれます。
2. **各表のプライマリーキーを 1 つ残します。** 検査一覧は `id_proc`、所見一覧は `id_item` です。
3. **1 セルには 1 つの値を入れます。** カンマで並べず、行か列を分けます。
4. **決まったカテゴリの列（肉眼型、病理、部位など）に、一覧にない値を入れません。** 集計で項目が分かれます。
5. **列の追加、行の削除、並べ替えはしてかまいません。** 削除した行数は記録します。`exclude` 列で除く方法もあります（[D.7](#exclude)）。
6. **列名は重ならないようにします。** スペース、改行、括弧は避けます。
7. **1 行目は列名、2 行目からはデータにします。**

## D.4 読み込んだあとの確認 {#check}

ここからは DuckDB の SQL と Python の polars の例です（タブで切り替え）。読み込んだら、まず D.3 が守られているかを確かめます。

=== "DuckDB SQL"

    ```sql
    CREATE TABLE proc AS SELECT * FROM read_csv('procedures.csv');  -- 検査一覧：1 行 1 検査
    CREATE TABLE item AS SELECT * FROM read_csv('items.csv');       -- 所見一覧：1 行 1 所見

    -- キーが一意か（結果が 0 行なら一意）
    SELECT id_proc, count(*) AS n FROM proc GROUP BY id_proc HAVING n > 1;
    SELECT id_item, count(*) AS n FROM item GROUP BY id_item HAVING n > 1;

    -- 所見の id_proc がすべて検査一覧にあるか（結果が 0 行なら OK）
    SELECT * FROM item WHERE id_proc NOT IN (SELECT id_proc FROM proc);

    -- カテゴリの列に一覧にない値がないか
    SELECT site, count(*) AS n FROM item
    WHERE site NOT IN ('rectum', 'sigmoid', 'descending', 'transverse', 'ascending', 'cecum')
    GROUP BY site;

    -- 型、数値の範囲、欠測の数。重なった列名には _1 がつく
    DESCRIBE item;
    SUMMARIZE item;
    ```

=== "polars"

    ```python
    import polars as pl

    proc = pl.read_csv("procedures.csv", try_parse_dates=True)  # 検査一覧：1 行 1 検査
    item = pl.read_csv("items.csv")                             # 所見一覧：1 行 1 所見

    # キーが一意か
    assert proc["id_proc"].is_unique().all()
    assert item["id_item"].is_unique().all()

    # 所見の id_proc がすべて検査一覧にあるか（0 行なら OK）
    print(item.filter(~pl.col("id_proc").is_in(proc["id_proc"].implode())))

    # 列名の重なり。polars は重なった列名に _duplicated_ をつけて読み込みます
    assert not any("_duplicated_" in c for c in item.columns)

    # カテゴリの列に一覧にない値がないか
    allowed = ["rectum", "sigmoid", "descending", "transverse", "ascending", "cecum"]
    print(item.filter(~pl.col("site").is_in(allowed))["site"].value_counts())

    # 型、数値の範囲、欠測の数
    print(item.schema)
    print(item.describe())
    ```

`size_mm` が数値でなく文字（`VARCHAR`、`str`）なら「15mm」や「不明」が、`date` が日付型でなければ書き方の違う日付が混ざっています。直すときは、何をどう直したかを残します。

## D.5 表を結合する {#join}

**所見一覧に検査の情報をつける**（所見ごとのミダゾラムの有無）

=== "DuckDB SQL"

    ```sql
    SELECT item.*, proc.midazolam
    FROM item
    LEFT JOIN proc USING (id_proc);
    ```

=== "polars"

    ```python
    item_with_proc = item.join(
        proc.select("id_proc", "midazolam"),
        on="id_proc",
        how="left",
        validate="m:1",   # 所見は多、検査は 1。検査一覧に同じ id_proc が 2 行あればエラー
    )
    ```

**検査一覧に所見の情報をつける**（検査ごとの浸潤癌の数。所見のない検査は 0）

=== "DuckDB SQL"

    ```sql
    SELECT proc.*, coalesce(n.n_invasive, 0) AS n_invasive
    FROM proc
    LEFT JOIN (
        SELECT id_proc, count(*) FILTER (WHERE pathology = 'invasive') AS n_invasive
        FROM item
        GROUP BY id_proc
    ) AS n USING (id_proc);
    ```

=== "polars"

    ```python
    n_invasive = item.group_by("id_proc").agg(
        (pl.col("pathology") == "invasive").sum().alias("n_invasive")
    )
    proc_with_item = proc.join(n_invasive, on="id_proc", how="left", validate="1:1").with_columns(
        pl.col("n_invasive").fill_null(0)
    )
    ```

キーが崩れていると、結合で行が増えたり別の検査の情報がついたりします。polars の `validate` はこれをエラーにします。SQL では D.4 のキーの確認を先に済ませ、結合の前後で行数を比べます。

## D.6 患者ごとの時系列 {#timeline}

患者 ID と日付があれば、患者ごとの経過を集計できます。例は、内視鏡的粘膜下層剥離術（endoscopic submucosal dissection、ESD）のあと 180 日以内に、内視鏡的バルーン拡張術（endoscopic balloon dilation、EBD）を受けたかどうかです。

=== "DuckDB SQL"

    ```sql
    SELECT esd.id_proc,
           coalesce(bool_or(ebd.date > esd.date
                            AND ebd.date <= esd.date + INTERVAL 180 DAY), false) AS ebd_180
    FROM proc AS esd
    LEFT JOIN proc AS ebd
        ON ebd.id_patient = esd.id_patient AND ebd.procedure = 'EBD'
    WHERE esd.procedure = 'ESD'
    GROUP BY esd.id_proc;
    ```

=== "polars"

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

`procedure` に「esd」や「ESD 」が混ざっていたり、日付が文字のままだったりすると、数え漏れます（[D.1](#example)）。生存時間解析の `time` と `event` も、同じように患者ごとの経過から作ります（[第 16 章](16_survival.md)）。

## D.7 行を除くとき {#exclude}

行を削除すると、何件をなぜ除いたかが残りません。論文では除外の数が要ります（[付録 B](B_reporting-guidelines.md) の STROBE など）。行は残し、`exclude` 列に理由を書いておくと便利です。

| id_proc | … | exclude |
|---------|---|---------|
| 1 | … | |
| 2 | … | 同意なし |
| 3 | … | 他院で治療済み |

=== "DuckDB SQL"

    ```sql
    SELECT exclude, count(*) AS n FROM proc GROUP BY exclude;   -- 除外の理由ごとの数
    CREATE TABLE analysis AS SELECT * FROM proc WHERE exclude IS NULL;
    ```

=== "polars"

    ```python
    print(proc["exclude"].value_counts())        # 除外の理由ごとの数
    analysis = proc.filter(pl.col("exclude").is_null())
    ```

理由ごとの数が、そのまま対象者の流れ図の数字になります。表が整えば、[第 1 章](01_descriptive.md#table1) の Table 1 から解析を始められます。

## 文献 {#references}

- Wickham H. Tidy data. Journal of Statistical Software 2014;59(10):1–23.
- Broman KW, Woo KH. Data organization in spreadsheets. The American Statistician 2018;72(1):2–10. 表計算ソフトでの注意がさらに詳しくまとまっています。

## まとめ

- 整然データは、1 行 1 観測、1 列 1 変数、1 セル 1 値の表です。
- 検査一覧と所見一覧は、プライマリーキー（`id_proc`、`id_item`）でつながります。識別列には手を触れません。
- 崩れた台帳の多くは、エラーが出ずに誤った数字になります。
- 読み込んだら、キー、カテゴリの値、型を確かめてから結合します。

**最後のページです。**[目次に戻る](index.md)
