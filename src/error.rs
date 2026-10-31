use thiserror::Error;

#[derive(Debug, Error)]
pub enum DataCollectionError {
    #[error("Cascade requires an 8x8 board, got side length {0}")]
    InvalidBoardSide(u8),
    #[error("board cell count must be 64, got {0}")]
    InvalidCellCount(usize),
    #[error("action index {0} is outside the valid range")]
    InvalidActionEncoding(u16),
    #[error("position ({row}, {col}) is out of bounds for an 8x8 board")]
    OutOfBounds { row: u8, col: u8 },
    #[error("stack height must be at least 1")]
    InvalidStackHeight,
    #[error("illegal action: {0}")]
    IllegalAction(String),
    #[error("selected action is not present in the recorded action statistics")]
    SelectedActionMissing,
    #[error("cannot attach a selected action to a terminal sample with no actions")]
    SelectedActionWithoutActions,
    #[error("invalid neural model format: {0}")]
    InvalidModelFormat(String),
    #[error("onnx runtime error: {0}")]
    OnnxRuntime(String),
    #[error("parallel execution error: {0}")]
    ParallelExecution(String),
    #[error("invalid configuration: {0}")]
    InvalidConfiguration(String),
    #[error(transparent)]
    RayonThreadPool(#[from] rayon::ThreadPoolBuildError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Serialization(#[from] serde_json::Error),
}
