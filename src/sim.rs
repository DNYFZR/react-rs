// Core Simulation Module
use crate::tx;

use polars::prelude::*;
use rand::{self, RngExt};
use rayon::prelude::*;

pub fn engine(
    n_sims: i64,
    n_steps: i64,
    uuids: Vec<String>,
    states: Vec<i64>,
    probabilities: Vec<Vec<f64>>,
    costs: Option<&Vec<i64>>,
    para_limit: i64,
) -> Result<DataFrame, PolarsError> {
    // Configrue parallel loops
    let batches = n_sims / para_limit;
    let remainder = n_sims - batches * para_limit;

    let loop_batches = match remainder {
        0 => batches,
        _ => batches + 1,
    };

    // Run
    let mut output: Vec<LazyFrame> = Vec::new();
    for batch in 0..loop_batches {
        let batch_size = if n_sims < para_limit {
            n_sims
        } else if batch > batches {
            remainder
        } else {
            para_limit
        };
        let sim_batch_id = batch * para_limit;

        // Run Sim
        let results = (0..batch_size)
            .into_par_iter()
            .map(|sim_id| {
                // Run sim
                let run_id: i64 = sim_batch_id + sim_id;
                return execute_event(run_id, &uuids, &states, &probabilities, &n_steps, costs)
                    .expect("failed to run simulation")
                    .lazy();
            })
            .collect::<Vec<LazyFrame>>();
        output.push(concat(results, UnionArgs::default())?);
    }

    Ok(concat(output, UnionArgs::default())?.collect()?)
}

fn execute_event(
    run_id: i64,
    uuids: &Vec<String>,
    states: &Vec<i64>,
    probabilities: &Vec<Vec<f64>>,
    n_steps: &i64,
    costs: Option<&Vec<i64>>,
) -> Result<DataFrame, PolarsError> {
    let costs_is_some = costs.is_some();

    let mut tmp_df = vec![Column::new(PlSmallStr::from_str("uuid"), uuids)];
    tmp_df.extend(
        discrete_event(states, probabilities, n_steps)
            .iter()
            .enumerate()
            .map(|(n, c)| Column::new(PlSmallStr::from_str(&format!("step_{n}")), c))
            .collect::<Vec<Column>>(),
    );

    let mut tmp_df =
        &mut DataFrame::new(tmp_df[0].len(), tmp_df).expect("failed to create table...");

    if costs_is_some {
        tmp_df = tmp_df.with_column(
            Series::from_vec(PlSmallStr::from_str("cost"), costs.unwrap().clone()).into_column(),
        )?;
    }

    // Add id column
    tmp_df = tmp_df.with_column(
        Series::from_vec(
            PlSmallStr::from_str("sim_id"),
            vec![run_id as i64; tmp_df.shape().0],
        )
        .into_column(),
    )?;

    Ok(tmp_df.clone())
}

fn discrete_event(
    states: &Vec<i64>,
    probabilities: &Vec<Vec<f64>>,
    n_steps: &i64,
) -> Vec<Vec<i64>> {
    // It is 2x faster to transpose the row results than run as columns
    let n_steps: usize = *n_steps as usize + 1;
    return tx::transpose(
        &states
            .into_par_iter()
            .enumerate()
            .map(|(idx, &v)| {
                let mut para_thrd: rand::rngs::SmallRng = rand::make_rng();

                let mut row: Vec<i64> = Vec::with_capacity(n_steps);
                row.push(v);

                let mut active_value = v;
                for _ in 1..n_steps {
                    let new_val = active_value + 1;
                    if let Some(curve) = probabilities.get(idx) {
                        if let Some(prob) = curve.get(new_val as usize) {
                            if prob > &para_thrd.random::<f64>() {
                                active_value = new_val;
                            } else {
                                active_value = 0;
                            }
                        } else {
                            active_value = 0;
                        }
                    } else {
                        panic!("Could not access survival curve at index = {} ", idx);
                    }

                    row.push(active_value);
                }
                return row;
            })
            .collect::<Vec<Vec<i64>>>(),
    );
}
