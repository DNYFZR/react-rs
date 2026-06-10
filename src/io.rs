// File I/O Module
use polars::prelude::*;

// Delta Tables
// pub async fn read_delta(table_uri: Url) -> Result<DeltaTable, DeltaTableError> {
//     let table = builder_from_valid_url(table_uri)?.load().await?;
//     Ok(table)
// }

// Parquet Files
// used for testing only atm
pub fn read_parquet(path: &str) -> PolarsResult<DataFrame> {
    Ok(ParquetReader::new(std::fs::File::open(path)?).finish()?)
}

#[test]
fn test_read_parquet() {
    // Ensure expected columns are in table
    let test = vec!["uuid", "step_0", "value"];
    let res: Vec<String> = read_parquet("./tests/data/demo_input.parquet")
        .expect("failed to read file")
        .get_column_names()
        .iter()
        .map(|s| s.to_string())
        .collect();

    assert_eq!(test, res);
}
