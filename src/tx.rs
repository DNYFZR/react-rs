// Data Transformation Module
use polars::prelude::*;
use rand::{self, RngExt};
use rayon::prelude::*;

pub fn constrain_parallel(
    table: &DataFrame,
    constrain_steps: i64,
    iter_regex: &str,
    cost_col: &str,
    limit_array: Vec<i64>,
    partition_by: &str,
    parallel_limit: i64,
) -> Result<DataFrame, PolarsError> {
    // Select only target iteration steps
    let target_steps: Vec<String> = (0..=constrain_steps)
        .into_par_iter()
        .map(|v| {
            if PlSmallStr::from_str(iter_regex).ends_with("_") {
                format!("{}{}", iter_regex, v)
            } else {
                format!("{}_{}", iter_regex, v)
            }
        })
        .collect();

    let select_cols: Vec<String> = table
        .get_column_names()
        .into_par_iter()
        .filter(|c| c.contains(iter_regex) == false || target_steps.contains(&c.to_string()))
        .map(|s| s.to_string())
        .collect();

    let table = &table.select(select_cols)?;

    // Create iteration map
    let active_cols: Vec<(usize, String)> = table
        .get_column_names()
        .iter()
        // We skip step 0 as the initial state cannot be altered
        .filter(|c| {
            c.contains(iter_regex) && ***c != PlSmallStr::from_str(&format!("{}_{}", iter_regex, 0))
        })
        .enumerate()
        .map(|(i, c)| (i + 1, c.to_string()))
        .collect();

    // Extract unique values from partition col
    let mut sim_ids: Vec<i64> = col_to_vec_i64(&table, partition_by);
    sim_ids.dedup();

    // Map partitions to vec
    let mut table_chunks: Vec<DataFrame> = vec![];
    let _ = sim_ids
        .iter()
        .map(|x| {
            let sim_df = table
                .clone()
                .lazy()
                .filter(col(partition_by).eq(lit(*x)))
                .collect()
                .ok()?;
            table_chunks.push(sim_df.clone());
            return Some(*x);
        })
        .collect::<Vec<Option<i64>>>();

    // configure parallel iterations
    let n_sims = table_chunks.len();
    let batches = n_sims / parallel_limit as usize;
    let remainder = n_sims - batches * parallel_limit as usize;
    let loop_batches = match remainder {
        0 => batches,
        _ => batches + 1,
    };

    // Process chunks in parallel
    let mut output: Vec<LazyFrame> = Vec::new();
    for batch in 0..loop_batches {
        let batch_size = if n_sims < parallel_limit as usize {
            n_sims
        } else if batch > batches {
            remainder
        } else {
            parallel_limit as usize
        };

        // Run batches of frame chunks in parallel
        let start_idx = batch * batch_size;
        let end_idx = start_idx + (batch_size - 1);

        // Should we change this to run over sim_id vec & filter table ???
        let processed_chunks = table_chunks[start_idx..=end_idx]
            .into_par_iter()
            .map(|chunk| {
                let mut df_sim = chunk.clone();
                let mut thrd: rand::rngs::SmallRng = rand::make_rng();
                let n_rows = df_sim.height();

                for (idx, col_name) in &active_cols {
                    // Add columns for random ordering & applied costs
                    df_sim = df_sim
                        .lazy()
                        .with_columns([
                            // Create randomised ordering column
                            lit(Series::from_vec(
                                PlSmallStr::from_str("order_col"),
                                (0..n_rows)
                                    .map(|_| thrd.random::<f64>())
                                    .collect::<Vec<f64>>(),
                            ))
                            .alias("order_col"),
                            // Identify when cost is realised
                            when(col(col_name).eq(lit(0)))
                                .then(col(cost_col))
                                .otherwise(lit(0))
                                .alias("applied_cost"),
                        ])
                        .collect()
                        .ok()?;

                    // Sort & totalise
                    df_sim = df_sim
                        .lazy()
                        .with_columns([col("applied_cost")
                            .cum_sum(true)
                            .sort_by(
                                ["order_col"],
                                SortMultipleOptions::new().with_order_descending(true),
                            )
                            .alias("totaliser")])
                        .collect()
                        .ok()?;

                    // Increase age if cost is above set limit
                    // use idx - 1 as we skip step 0 (init state)
                    let step_limit = lit(limit_array[*idx - 1 as usize] as f64);
                    df_sim = df_sim
                        .lazy()
                        .with_columns([when(
                            col("totaliser")
                                .lt(step_limit)
                                .and(col(col_name).eq(lit(0 as i64))),
                        )
                        .then(col(col_name))
                        .otherwise(col(&format!("step_{}", idx - 1)) + lit(1))
                        .alias("tmp_col")])
                        .collect()
                        .ok()?;

                    // Update following years when event is deferred
                    for (i, c) in active_cols.iter().rev() {
                        if i > idx {
                            df_sim = df_sim
                                .lazy()
                                .with_columns([when(col("tmp_col").neq(col(col_name)))
                                    .then(col(&format!("step_{}", i - 1)))
                                    .otherwise(col(c))
                                    .alias(c)])
                                .collect()
                                .ok()?;
                        }
                    }

                    // Update current year
                    df_sim = df_sim
                        .lazy()
                        .with_columns([col("tmp_col").alias(col_name)])
                        .collect()
                        .ok()?;

                    // Remove process cols
                    df_sim =
                        df_sim.drop_many(["order_col", "applied_cost", "totaliser", "tmp_col"]);
                }

                return Some(df_sim.lazy());
            })
            .collect::<Vec<Option<LazyFrame>>>();

        // Extract & combine batches
        let processed_chunks = processed_chunks
            .iter()
            .map(|df| {
                if let Some(df) = df {
                    df.clone()
                } else {
                    // this is to satisfy Rust compiler & allow us to unwrap the Option<LazyFrame)
                    // there will always be Some(df) as it is a processed chunk of the user input
                    DataFrame::empty().lazy()
                }
            })
            .collect::<Vec<LazyFrame>>();
        output.push(concat(processed_chunks, UnionArgs::default())?);
    }

    // Combine all chunks into single df
    return Ok(concat(output, UnionArgs::default())?.collect()?);
}

pub fn transpose(v: &Vec<Vec<i64>>) -> Vec<Vec<i64>> {
    let wid = v[0].len();

    (0..wid)
        .into_par_iter()
        .map(|i| v.iter().map(|row| row[i]).collect())
        .collect()
}

pub fn col_to_vec_i64(df: &DataFrame, col: &str) -> Vec<i64> {
    return df
        .column(col)
        .expect("failed to get col...")
        .i64()
        .expect("failed to get i64 array")
        .into_iter()
        .map(|v| v.unwrap())
        .collect();
}

pub fn array_col_to_vec_f64(df: &DataFrame, col: &str) -> Vec<Vec<f64>> {
    let arr = df.column(col).unwrap().list().unwrap();

    let mut out = Vec::with_capacity(arr.len());

    for opt_list in arr.into_iter() {
        let list = opt_list
            .ok_or_else(|| PolarsError::ComputeError("null in array column".into()))
            .unwrap();
        let vals = list
            .f64()
            .unwrap()
            .into_no_null_iter()
            .collect::<Vec<f64>>();
        out.push(vals);
    }

    return out;
}

pub fn col_to_vec_str(df: &DataFrame, col: &str) -> Vec<String> {
    return df
        .column(col)
        .expect("failed to get col...")
        .str()
        .expect("failed to get i64 array")
        .into_iter()
        .map(|v| String::from(v.unwrap()))
        .collect();
}
