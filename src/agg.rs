// Aggregation Module
use crate::tx::col_to_vec_i64;
use polars::prelude::*;
use rayon::prelude::*;

pub fn aggregate(
    df: DataFrame,
    partition_by: &str,
    iter_regex: &str,
    target_value: i64,
    cost_col: Option<&str>,
) -> Result<DataFrame, PolarsError> {
    let converted_table = match cost_col {
        Some(cost_col) => convert(&df, iter_regex, &target_value, Some(cost_col)),
        None => convert(&df, iter_regex, &target_value, None),
    };

    match converted_table {
        Ok(table) => return aggregate_event(table, partition_by, iter_regex),
        Err(e) => return Err(e),
    };
}

fn aggregate_event(
    table: DataFrame,
    partition_by: &str,
    iter_regex: &str,
) -> Result<DataFrame, PolarsError> {
    // Get non-id cols & skip starting age
    let initial_step = if iter_regex.ends_with("_") {
        &format!("{}{}", iter_regex, 0)
    } else {
        &format!("{}_{}", iter_regex, 0)
    };

    let agg_cols: Vec<String> = table
        .get_column_names()
        .into_par_iter()
        .filter(|s| s.to_string().contains(iter_regex) & s.to_string().ne(initial_step))
        .map(|v| v.to_string())
        .collect();

    return Ok(table
        .lazy()
        .drop(Selector::ByName {
            names: Arc::from([PlSmallStr::from_str("cost")]),
            strict: false,
        })
        .group_by([partition_by])
        .agg(
            agg_cols
                .into_par_iter()
                .map(|c| col(c.clone()).sum().alias(c))
                .collect::<Vec<Expr>>(),
        )
        .collect()?);
}

fn convert(
    table: &DataFrame,
    iter_regex: &str,
    target_value: &i64,
    cost_col: Option<&str>,
) -> Result<DataFrame, PolarsError> {
    // Get non-id cols
    let agg_cols: Vec<String> = table
        .get_column_names()
        .into_par_iter()
        .filter(|&s| s.contains(iter_regex))
        .map(|v| v.to_string())
        .collect();

    // Run conversion
    return Ok(table
        .clone()
        .lazy()
        .with_columns(
            agg_cols
                .into_par_iter()
                .map(|col_name| match cost_col {
                    Some(cost_col) => {
                        return (when(col(col_name.clone()).eq(*target_value))
                            .then(col(cost_col))
                            .otherwise(lit(0 as i64)))
                        .alias(col_name);
                    }
                    None => {
                        return (when(col(col_name.clone()).eq(*target_value))
                            .then(lit(1 as i64))
                            .otherwise(lit(0 as i64)))
                        .alias(col_name);
                    }
                })
                .collect::<Vec<Expr>>(),
        )
        .collect()?);
}

pub fn count_values(
    table: &DataFrame,
    partition_by: &str,
    iter_regex: &str,
) -> Result<DataFrame, PolarsError> {
    let mut sim_ids = col_to_vec_i64(&table, partition_by);
    sim_ids.dedup();

    let mut sim_results = vec![];
    for sim_id in sim_ids {
        let container = table
            .get_column_names()
            .into_par_iter()
            .filter(|c| c.contains(iter_regex))
            .map(|c| {
                let val_counts = table
                    .select(vec![c])
                    .unwrap()
                    .rename(c, PlSmallStr::from_str("value"))
                    .unwrap()
                    .column("value")
                    .unwrap()
                    .as_series()
                    .unwrap()
                    .value_counts(false, false, PlSmallStr::from_str(c), false)
                    .expect("failed to count values");

                return val_counts
                    .clone()
                    .lazy()
                    .with_column(lit(sim_id).alias(partition_by))
                    .select([col(partition_by), col("value"), col(c.to_string())]);
            })
            .collect::<Vec<LazyFrame>>();

        // update table
        let mut df = container[0].clone();
        for idx in 1..container.len() {
            df = df.lazy().join(
                container[idx].clone(),
                [col(partition_by), col("value")],
                [col(partition_by), col("value")],
                JoinArgs::new(JoinType::Left),
            );
        }
        sim_results.push(df);
    }

    // Combine results
    let df = concat(sim_results, UnionArgs::default())?;

    // Replace nulls with zero
    return Ok(df
        .fill_null(0)
        .sort([partition_by, "value"], SortMultipleOptions::default())
        .collect()
        .expect("failed to sort..."));
}
