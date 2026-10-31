use crate::engine::GameState;
use crate::eval::{terminal_value_for_player, Evaluation, PositionEvaluator};
use crate::error::DataCollectionError;
use crate::model::{Action, Direction, Phase, Player, Position, BOARD_CELLS, BOARD_SIDE, TOTAL_ACTIONS};
use libm::erff;
use ort::{
    CUDAExecutionProvider, ROCmExecutionProvider,
    GraphOptimizationLevel, Session,
    Tensor,
};
use serde::Deserialize;
use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

pub const V1_CELL_FEATURE_DIM: usize = 6;
pub const V1_GLOBAL_FEATURE_DIM: usize = 10;
pub const V2_CELL_FEATURE_DIM: usize = 14;
pub const V2_GLOBAL_FEATURE_DIM: usize = 24;
pub const V1_INPUT_DIM: usize = BOARD_CELLS * V1_CELL_FEATURE_DIM + V1_GLOBAL_FEATURE_DIM;
pub const V2_INPUT_DIM: usize = BOARD_CELLS * V2_CELL_FEATURE_DIM + V2_GLOBAL_FEATURE_DIM;
pub const TEACHER_FEATURE_PLANES: usize = 32;

const MAX_STACK_HEIGHT: f32 = 12.0;
const MAX_PIECES_PER_PLAYER: f32 = 12.0;
const MAX_STACKS_PER_PLAYER: f32 = 12.0;
const MAX_REPETITIONS: f32 = 3.0;
const MAX_PLAY_TURN: f32 = 300.0;
const MAX_PLY: f32 = 400.0;
const SQRT_2: f32 = std::f32::consts::SQRT_2;
const DEFAULT_LAYER_NORM_EPS: f32 = 1e-5;
const DEFAULT_TEACHER_INPUT_NAME: &str = "input";
const DEFAULT_TEACHER_POLICY_OUTPUT_NAME: &str = "policy_logits";
const DEFAULT_TEACHER_VALUE_OUTPUT_NAME: &str = "value";

static ORT_RUNTIME_INIT: OnceLock<Result<Option<PathBuf>, String>> = OnceLock::new();

const LOCAL_SUPPORT_SCALE: f32 = 5.5;
const LOCAL_ATTACK_SCALE: f32 = 6.0;
const LOCAL_VULNERABILITY_SCALE: f32 = 6.0;
const LOCAL_CASCADE_SCALE: f32 = 4.0;
const GLOBAL_ATTACK_PRESSURE_SCALE: f32 = 24.0;
const GLOBAL_VULNERABLE_MASS_SCALE: f32 = 48.0;
const GLOBAL_SUPPORT_SCALE: f32 = 12.0;
const GLOBAL_CENTER_MASS_SCALE: f32 = 8.0;
const GLOBAL_EDGE_EXPOSURE_SCALE: f32 = 12.0;
const GLOBAL_STACK_PROFILE_SCALE: f32 = 3.0;
const GLOBAL_CASCADE_POTENTIAL_SCALE: f32 = 12.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelFormat {
    V1,
    V2,
}

impl ModelFormat {
    fn parse(name: &str) -> Result<Self, DataCollectionError> {
        match name {
            "cascade_student_mlp_v1" => Ok(Self::V1),
            "cascade_student_mlp_v2" => Ok(Self::V2),
            _ => Err(DataCollectionError::InvalidModelFormat(format!(
                "unsupported manifest format: {name}"
            ))),
        }
    }

    const fn layout(self) -> FeatureLayout {
        match self {
            Self::V1 => FeatureLayout {
                cell_feature_dim: V1_CELL_FEATURE_DIM,
                global_feature_dim: V1_GLOBAL_FEATURE_DIM,
                action_dim: TOTAL_ACTIONS,
            },
            Self::V2 => FeatureLayout {
                cell_feature_dim: V2_CELL_FEATURE_DIM,
                global_feature_dim: V2_GLOBAL_FEATURE_DIM,
                action_dim: TOTAL_ACTIONS,
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FeatureLayout {
    pub cell_feature_dim: usize,
    pub global_feature_dim: usize,
    pub action_dim: usize,
}

impl FeatureLayout {
    pub const fn new() -> Self {
        ModelFormat::V2.layout()
    }

    pub const fn input_dim(self) -> usize {
        self.cell_feature_dim * BOARD_CELLS + self.global_feature_dim
    }
}

impl Default for FeatureLayout {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TeacherFeatureLayoutVersion {
    #[default]
    LegacyV1,
    HeightBucketsV2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TeacherFeaturePlaneLayout {
    pub plane_count: usize,
    pub board_side: usize,
    pub action_dim: usize,
    pub version: TeacherFeatureLayoutVersion,
}

impl TeacherFeaturePlaneLayout {
    pub const fn new() -> Self {
        Self::with_version(TeacherFeatureLayoutVersion::LegacyV1)
    }

    pub const fn with_version(version: TeacherFeatureLayoutVersion) -> Self {
        Self {
            plane_count: TEACHER_FEATURE_PLANES,
            board_side: BOARD_SIDE as usize,
            action_dim: TOTAL_ACTIONS,
            version,
        }
    }

    pub const fn input_dim(self) -> usize {
        self.plane_count * BOARD_CELLS
    }
}

impl Default for TeacherFeaturePlaneLayout {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct NeuralEvaluatorConfig {
    pub policy_temperature: f32,
}

impl NeuralEvaluatorConfig {
    pub fn with_policy_temperature(mut self, policy_temperature: f32) -> Self {
        self.policy_temperature = policy_temperature.max(0.05);
        self
    }
}

impl Default for NeuralEvaluatorConfig {
    fn default() -> Self {
        Self {
            policy_temperature: 1.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct NeuralEvaluator {
    config: NeuralEvaluatorConfig,
    model: NeuralModel,
}

impl NeuralEvaluator {
    pub fn load<P>(manifest_path: P, config: NeuralEvaluatorConfig) -> Result<Self, DataCollectionError>
    where
        P: AsRef<Path>,
    {
        let manifest_text = fs::read_to_string(manifest_path.as_ref())?;
        let format_probe: FormatProbe = serde_json::from_str(&manifest_text)?;
        let model = match format_probe.format.as_str() {
            "cascade_teacher_onnx_v1" => NeuralModel::Teacher(TeacherResNetModel::load(
                manifest_path.as_ref(),
                &manifest_text,
            )?),
            _ => NeuralModel::Student(StudentMlpModel::load(
                manifest_path.as_ref(),
                &manifest_text,
            )?),
        };
        Ok(Self { config, model })
    }

    pub fn feature_layout(&self) -> FeatureLayout {
        match &self.model {
            NeuralModel::Student(model) => model.layout,
            NeuralModel::Teacher(_) => FeatureLayout::new(),
        }
    }

    pub fn teacher_feature_layout(&self) -> Option<TeacherFeaturePlaneLayout> {
        match &self.model {
            NeuralModel::Student(_) => None,
            NeuralModel::Teacher(model) => Some(model.layout),
        }
    }
}

impl PositionEvaluator for NeuralEvaluator {
    fn evaluate(&self, state: &GameState) -> Result<Evaluation, DataCollectionError> {
        if let Some(status) = state.terminal_status() {
            return Ok(Evaluation {
                value: terminal_value_for_player(status, state.turn()),
                priors: Vec::new(),
            });
        }

        let legal_actions = state.legal_actions();
        let (policy_logits, value) = match &self.model {
            NeuralModel::Student(model) => {
                let features = encode_state_features(state, model.format)?;
                model.forward(&features)?
            }
            NeuralModel::Teacher(model) => {
                let features = encode_teacher_feature_planes_with_version(state, model.layout.version)?;
                model.forward(&features)?
            }
        };

        if legal_actions.is_empty() {
            return Ok(Evaluation {
                value,
                priors: Vec::new(),
            });
        }

        let priors = masked_policy_from_logits(
            &legal_actions,
            &policy_logits,
            self.config.policy_temperature,
            state.turn(),
        );
        Ok(Evaluation { value, priors })
    }
}

#[derive(Debug, Clone)]
enum NeuralModel {
    Student(StudentMlpModel),
    Teacher(TeacherResNetModel),
}

#[derive(Debug, Clone, Deserialize)]
struct FormatProbe {
    format: String,
}

#[derive(Debug, Clone)]
struct StudentMlpModel {
    format: ModelFormat,
    layout: FeatureLayout,
    trunk: Vec<MlpBlock>,
    policy_head: LinearLayer,
    value_hidden: LinearLayer,
    value_out: LinearLayer,
    layer_norm_eps: f32,
}

impl StudentMlpModel {
    fn load(manifest_path: &Path, manifest_text: &str) -> Result<Self, DataCollectionError> {
        let manifest: ModelManifest = serde_json::from_str(manifest_text)?;
        let format = ModelFormat::parse(&manifest.format)?;
        validate_manifest(format, &manifest)?;

        let weights_path = resolve_artifact_path(manifest_path, &manifest.weights_file);
        let raw_bytes = fs::read(weights_path)?;
        let float_data = bytes_to_f32(&raw_bytes)?;
        let tensor_specs: HashMap<_, _> = manifest
            .tensors
            .iter()
            .map(|tensor| (tensor.name.as_str(), tensor))
            .collect();

        let layout = FeatureLayout {
            cell_feature_dim: manifest.cell_feature_dim,
            global_feature_dim: manifest.global_feature_dim,
            action_dim: manifest.action_dim,
        };

        let mut previous_dim = manifest.input_dim;
        let mut trunk = Vec::with_capacity(manifest.hidden_dims.len());
        for (index, &hidden_dim) in manifest.hidden_dims.iter().enumerate() {
            let linear_weight = tensor_vec(
                &float_data,
                &tensor_specs,
                &format!("trunk.{index}.linear.weight"),
                &[hidden_dim, previous_dim],
            )?;
            let linear_bias = tensor_vec(
                &float_data,
                &tensor_specs,
                &format!("trunk.{index}.linear.bias"),
                &[hidden_dim],
            )?;
            let norm_weight = tensor_vec(
                &float_data,
                &tensor_specs,
                &format!("trunk.{index}.norm.weight"),
                &[hidden_dim],
            )?;
            let norm_bias = tensor_vec(
                &float_data,
                &tensor_specs,
                &format!("trunk.{index}.norm.bias"),
                &[hidden_dim],
            )?;
            trunk.push(MlpBlock {
                linear: LinearLayer::new(previous_dim, hidden_dim, linear_weight, linear_bias)?,
                norm_weight,
                norm_bias,
            });
            previous_dim = hidden_dim;
        }

        let policy_head = LinearLayer::new(
            previous_dim,
            manifest.action_dim,
            tensor_vec(
                &float_data,
                &tensor_specs,
                "policy_head.weight",
                &[manifest.action_dim, previous_dim],
            )?,
            tensor_vec(
                &float_data,
                &tensor_specs,
                "policy_head.bias",
                &[manifest.action_dim],
            )?,
        )?;
        let value_hidden = LinearLayer::new(
            previous_dim,
            manifest.value_hidden_dim,
            tensor_vec(
                &float_data,
                &tensor_specs,
                "value_head.0.weight",
                &[manifest.value_hidden_dim, previous_dim],
            )?,
            tensor_vec(
                &float_data,
                &tensor_specs,
                "value_head.0.bias",
                &[manifest.value_hidden_dim],
            )?,
        )?;
        let value_out = LinearLayer::new(
            manifest.value_hidden_dim,
            1,
            tensor_vec(
                &float_data,
                &tensor_specs,
                "value_head.3.weight",
                &[1, manifest.value_hidden_dim],
            )?,
            tensor_vec(
                &float_data,
                &tensor_specs,
                "value_head.3.bias",
                &[1],
            )?,
        )?;

        Ok(Self {
            format,
            layout,
            trunk,
            policy_head,
            value_hidden,
            value_out,
            layer_norm_eps: manifest.layer_norm_eps.unwrap_or(DEFAULT_LAYER_NORM_EPS),
        })
    }

    fn forward(&self, input: &[f32]) -> Result<(Vec<f32>, f32), DataCollectionError> {
        if self.layout.input_dim() != input.len() {
            return Err(DataCollectionError::InvalidModelFormat(format!(
                "model input_dim {} does not match provided input {}",
                self.layout.input_dim(),
                input.len(),
            )));
        }

        let mut current = input.to_vec();
        let mut scratch = Vec::new();
        for block in &self.trunk {
            scratch.resize(block.linear.out_dim, 0.0);
            block.linear.forward_into(&current, &mut scratch)?;
            layer_norm_in_place(&mut scratch, &block.norm_weight, &block.norm_bias, self.layer_norm_eps)?;
            gelu_in_place(&mut scratch);
            std::mem::swap(&mut current, &mut scratch);
        }

        let mut policy_logits = vec![0.0; self.policy_head.out_dim];
        self.policy_head.forward_into(&current, &mut policy_logits)?;

        let mut value_hidden = vec![0.0; self.value_hidden.out_dim];
        self.value_hidden.forward_into(&current, &mut value_hidden)?;
        gelu_in_place(&mut value_hidden);

        let mut value_out = [0.0f32; 1];
        self.value_out.forward_into(&value_hidden, &mut value_out)?;
        Ok((policy_logits, value_out[0].tanh().clamp(-1.0, 1.0)))
    }
}

#[derive(Debug, Clone)]
struct TeacherResNetModel {
    layout: TeacherFeaturePlaneLayout,
    session: Arc<std::sync::Mutex<Session>>,
    input_name: String,
    policy_output_name: String,
    value_output_name: String,
}

impl TeacherResNetModel {
    fn load(manifest_path: &Path, manifest_text: &str) -> Result<Self, DataCollectionError> {
        let manifest: TeacherOnnxManifest = serde_json::from_str(manifest_text)?;
        validate_teacher_manifest(&manifest)?;
        ensure_ort_runtime(&manifest)?;

        let model_path = resolve_artifact_path(manifest_path, &manifest.onnx_file);
        let session_raw = build_teacher_session(&model_path, &manifest)?;
        validate_teacher_session_outputs(&session_raw, &manifest)?;
        let session = Arc::new(std::sync::Mutex::new(session_raw));

        Ok(Self {
            layout: TeacherFeaturePlaneLayout::with_version(manifest.feature_layout_version),
            session,
            input_name: manifest.input_name,
            policy_output_name: manifest.policy_output_name,
            value_output_name: manifest.value_output_name,
        })
    }

    fn forward(&self, input: &[f32]) -> Result<(Vec<f32>, f32), DataCollectionError> {
        if self.layout.input_dim() != input.len() {
            return Err(DataCollectionError::InvalidModelFormat(format!(
                "teacher model input_dim {} does not match provided input {}",
                self.layout.input_dim(),
                input.len(),
            )));
        }

        let input_tensor = Tensor::<f32>::from_array((
            [
                1usize,
                self.layout.plane_count,
                self.layout.board_side,
                self.layout.board_side,
            ],
            input.to_vec(),
        ))
        .map_err(|error| DataCollectionError::OnnxRuntime(error.to_string()))?;
        let inputs = ort::inputs! {
            self.input_name.as_str() => input_tensor
        }
        .map_err(|error| DataCollectionError::OnnxRuntime(error.to_string()))?;
        let guard = self.session.lock().unwrap();
        let outputs = guard
            .run(inputs)
            .map_err(|error| DataCollectionError::OnnxRuntime(error.to_string()))?;

        let policy_output = outputs
            .get(self.policy_output_name.as_str())
            .ok_or_else(|| {
                DataCollectionError::InvalidModelFormat(format!(
                    "missing policy output {} in ONNX session",
                    self.policy_output_name
                ))
            })?;
        let (_, policy_logits) = policy_output
            .try_extract_raw_tensor::<f32>()
            .map_err(|error| DataCollectionError::OnnxRuntime(error.to_string()))?;
        if policy_logits.len() != self.layout.action_dim {
            return Err(DataCollectionError::InvalidModelFormat(format!(
                "policy output {} produced {} values, expected {}",
                self.policy_output_name,
                policy_logits.len(),
                self.layout.action_dim,
            )));
        }

        let value_output = outputs
            .get(self.value_output_name.as_str())
            .ok_or_else(|| {
                DataCollectionError::InvalidModelFormat(format!(
                    "missing value output {} in ONNX session",
                    self.value_output_name
                ))
            })?;
        let (_, value_tensor) = value_output
            .try_extract_raw_tensor::<f32>()
            .map_err(|error| DataCollectionError::OnnxRuntime(error.to_string()))?;
        if value_tensor.len() != 1 {
            return Err(DataCollectionError::InvalidModelFormat(format!(
                "value output {} produced {} values, expected 1",
                self.value_output_name,
                value_tensor.len(),
            )));
        }

        Ok((policy_logits.to_vec(), value_tensor[0].clamp(-1.0, 1.0)))
    }
}

#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum TeacherExecutionProvider {
    Cpu,
    #[default]
    Cuda,
    Rocm,
}

#[derive(Debug, Clone, Deserialize)]
struct TeacherOnnxManifest {
    format: String,
    onnx_file: String,
    #[serde(default)]
    feature_layout_version: TeacherFeatureLayoutVersion,
    feature_planes: usize,
    board_side: usize,
    action_dim: usize,
    #[serde(default = "default_teacher_input_name")]
    input_name: String,
    #[serde(default = "default_teacher_policy_output_name")]
    policy_output_name: String,
    #[serde(default = "default_teacher_value_output_name")]
    value_output_name: String,
    #[serde(default)]
    execution_provider: TeacherExecutionProvider,
    #[serde(default)]
    device_id: i32,
    #[serde(default)]
    allow_cpu_fallback: bool,
    #[serde(default)]
    runtime_library_path: Option<String>,
}

fn default_teacher_input_name() -> String {
    DEFAULT_TEACHER_INPUT_NAME.to_string()
}

fn default_teacher_policy_output_name() -> String {
    DEFAULT_TEACHER_POLICY_OUTPUT_NAME.to_string()
}

fn default_teacher_value_output_name() -> String {
    DEFAULT_TEACHER_VALUE_OUTPUT_NAME.to_string()
}

fn ensure_ort_runtime(manifest: &TeacherOnnxManifest) -> Result<(), DataCollectionError> {
    let requested_path = manifest
        .runtime_library_path
        .as_ref()
        .map(PathBuf::from)
        .or_else(|| env::var_os("ORT_DYLIB_PATH").map(PathBuf::from));
    let init_result = ORT_RUNTIME_INIT.get_or_init(|| {
        if let Some(path) = requested_path.as_deref() {
            ort::init_from(path.to_string_lossy().into_owned())
                .commit()
                .map(|_| requested_path.clone())
                .map_err(|error| error.to_string())
        } else {
            ort::init()
                .commit()
                .map(|_| None)
                .map_err(|error| error.to_string())
        }
    });

    match init_result {
        Ok(initialized_path) => {
            if let Some(requested) = requested_path.as_ref() {
                match initialized_path.as_ref() {
                    Some(initialized) if initialized == requested => {}
                    Some(initialized) => {
                        return Err(DataCollectionError::OnnxRuntime(format!(
                            "ONNX Runtime was already initialized with {:?}, cannot reinitialize with {:?}",
                            initialized, requested,
                        )));
                    }
                    None => {
                        return Err(DataCollectionError::OnnxRuntime(format!(
                            "ONNX Runtime was already initialized without an explicit dynamic library path, cannot reinitialize with {:?}",
                            requested,
                        )));
                    }
                }
            }
            Ok(())
        }
        Err(error) => Err(DataCollectionError::OnnxRuntime(error.clone())),
    }
}

fn build_teacher_session(
    model_path: &Path,
    manifest: &TeacherOnnxManifest,
) -> Result<Session, DataCollectionError> {
    let mut builder = Session::builder()
        .map_err(|error| DataCollectionError::OnnxRuntime(error.to_string()))?;
    builder = builder
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|error| DataCollectionError::OnnxRuntime(error.to_string()))?;
    builder = builder
        .with_intra_threads(1)
        .map_err(|error| DataCollectionError::OnnxRuntime(error.to_string()))?;

    if manifest.execution_provider == TeacherExecutionProvider::Cuda {
        let cuda_provider = CUDAExecutionProvider::default()
            .with_device_id(manifest.device_id)
            .build();
        builder = builder
            .with_execution_providers([if manifest.allow_cpu_fallback {
                cuda_provider
            } else {
                cuda_provider.error_on_failure()
            }])
            .map_err(|error| DataCollectionError::OnnxRuntime(error.to_string()))?;
    } else if manifest.execution_provider == TeacherExecutionProvider::Rocm {
        let rocm_provider = ROCmExecutionProvider::default()
            .with_device_id(manifest.device_id)
            .build();
        builder = builder
            .with_execution_providers([if manifest.allow_cpu_fallback {
                rocm_provider
            } else {
                rocm_provider.error_on_failure()
            }])
            .map_err(|error| DataCollectionError::OnnxRuntime(error.to_string()))?;
    }

    builder
        .commit_from_file(model_path)
        .map_err(|error| DataCollectionError::OnnxRuntime(error.to_string()))
}

fn validate_teacher_session_outputs(
    session: &Session,
    manifest: &TeacherOnnxManifest,
) -> Result<(), DataCollectionError> {
    if !session
        .inputs
        .iter()
        .any(|input| input.name == manifest.input_name.as_str())
    {
        return Err(DataCollectionError::InvalidModelFormat(format!(
            "ONNX model is missing input {}",
            manifest.input_name
        )));
    }

    if !session
        .outputs
        .iter()
        .any(|output| output.name == manifest.policy_output_name.as_str())
    {
        return Err(DataCollectionError::InvalidModelFormat(format!(
            "ONNX model is missing policy output {}",
            manifest.policy_output_name
        )));
    }

    if !session
        .outputs
        .iter()
        .any(|output| output.name == manifest.value_output_name.as_str())
    {
        return Err(DataCollectionError::InvalidModelFormat(format!(
            "ONNX model is missing value output {}",
            manifest.value_output_name
        )));
    }

    Ok(())
}

#[derive(Debug, Clone)]
struct MlpBlock {
    linear: LinearLayer,
    norm_weight: Vec<f32>,
    norm_bias: Vec<f32>,
}

#[derive(Debug, Clone)]
struct LinearLayer {
    in_dim: usize,
    out_dim: usize,
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl LinearLayer {
    fn new(in_dim: usize, out_dim: usize, weight: Vec<f32>, bias: Vec<f32>) -> Result<Self, DataCollectionError> {
        if weight.len() != in_dim * out_dim {
            return Err(DataCollectionError::InvalidModelFormat(format!(
                "linear weight length {} does not match shape [{out_dim}, {in_dim}]",
                weight.len()
            )));
        }
        if bias.len() != out_dim {
            return Err(DataCollectionError::InvalidModelFormat(format!(
                "linear bias length {} does not match output dim {out_dim}",
                bias.len()
            )));
        }

        Ok(Self {
            in_dim,
            out_dim,
            weight,
            bias,
        })
    }

    fn forward_into<T>(&self, input: &[f32], output: &mut T) -> Result<(), DataCollectionError>
    where
        T: AsMut<[f32]> + ?Sized,
    {
        let output = output.as_mut();
        if input.len() != self.in_dim || output.len() != self.out_dim {
            return Err(DataCollectionError::InvalidModelFormat(format!(
                "linear forward shape mismatch: input={} output={} expected input={} output={}",
                input.len(),
                output.len(),
                self.in_dim,
                self.out_dim,
            )));
        }

        for (row_index, row) in self.weight.chunks_exact(self.in_dim).enumerate() {
            let mut sum = self.bias[row_index];
            for (weight, value) in row.iter().zip(input) {
                sum += *weight * *value;
            }
            output[row_index] = sum;
        }

        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize)]
struct ModelManifest {
    format: String,
    weights_file: String,
    input_dim: usize,
    cell_feature_dim: usize,
    global_feature_dim: usize,
    action_dim: usize,
    hidden_dims: Vec<usize>,
    value_hidden_dim: usize,
    layer_norm_eps: Option<f32>,
    tensors: Vec<TensorManifest>,
}

#[derive(Debug, Clone, Deserialize)]
struct TensorManifest {
    name: String,
    shape: Vec<usize>,
    offset_floats: usize,
    length_floats: usize,
}

fn validate_manifest(format: ModelFormat, manifest: &ModelManifest) -> Result<(), DataCollectionError> {
    let expected_layout = format.layout();
    if manifest.cell_feature_dim != expected_layout.cell_feature_dim
        || manifest.global_feature_dim != expected_layout.global_feature_dim
        || manifest.input_dim != expected_layout.input_dim()
    {
        return Err(DataCollectionError::InvalidModelFormat(format!(
            "manifest layout ({}, {}, {}) does not match expected layout ({}, {}, {})",
            manifest.cell_feature_dim,
            manifest.global_feature_dim,
            manifest.input_dim,
            expected_layout.cell_feature_dim,
            expected_layout.global_feature_dim,
            expected_layout.input_dim(),
        )));
    }

    if manifest.action_dim != TOTAL_ACTIONS {
        return Err(DataCollectionError::InvalidModelFormat(format!(
            "manifest action_dim {} does not match expected {TOTAL_ACTIONS}",
            manifest.action_dim
        )));
    }

    Ok(())
}

fn validate_teacher_manifest(manifest: &TeacherOnnxManifest) -> Result<(), DataCollectionError> {
    if manifest.format != "cascade_teacher_onnx_v1" {
        return Err(DataCollectionError::InvalidModelFormat(format!(
            "unsupported teacher manifest format: {}",
            manifest.format
        )));
    }

    if manifest.feature_planes != TEACHER_FEATURE_PLANES {
        return Err(DataCollectionError::InvalidModelFormat(format!(
            "teacher manifest feature_planes {} does not match expected {}",
            manifest.feature_planes, TEACHER_FEATURE_PLANES,
        )));
    }

    if manifest.board_side != BOARD_SIDE as usize {
        return Err(DataCollectionError::InvalidModelFormat(format!(
            "teacher manifest board_side {} does not match expected {}",
            manifest.board_side, BOARD_SIDE,
        )));
    }

    if manifest.action_dim != TOTAL_ACTIONS {
        return Err(DataCollectionError::InvalidModelFormat(format!(
            "teacher manifest action_dim {} does not match expected {TOTAL_ACTIONS}",
            manifest.action_dim,
        )));
    }

    if manifest.onnx_file.trim().is_empty() {
        return Err(DataCollectionError::InvalidModelFormat(
            "teacher manifest onnx_file cannot be empty".to_string(),
        ));
    }

    Ok(())
}
fn resolve_artifact_path(manifest_path: &Path, artifact_file: &str) -> PathBuf {
    let artifact_path = Path::new(artifact_file);
    if artifact_path.is_absolute() {
        artifact_path.to_path_buf()
    } else {
        manifest_path.parent().unwrap_or_else(|| Path::new(".")).join(artifact_path)
    }
}

fn bytes_to_f32(raw_bytes: &[u8]) -> Result<Vec<f32>, DataCollectionError> {
    if raw_bytes.len() % 4 != 0 {
        return Err(DataCollectionError::InvalidModelFormat(format!(
            "weights file length {} is not a multiple of 4 bytes",
            raw_bytes.len()
        )));
    }

    Ok(raw_bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn tensor_vec(
    float_data: &[f32],
    tensor_specs: &HashMap<&str, &TensorManifest>,
    name: &str,
    expected_shape: &[usize],
) -> Result<Vec<f32>, DataCollectionError> {
    let spec = tensor_specs
        .get(name)
        .copied()
        .ok_or_else(|| DataCollectionError::InvalidModelFormat(format!("missing tensor {name}")))?;

    if spec.shape != expected_shape {
        return Err(DataCollectionError::InvalidModelFormat(format!(
            "tensor {name} shape {:?} does not match expected {:?}",
            spec.shape, expected_shape
        )));
    }

    let end = spec.offset_floats.checked_add(spec.length_floats).ok_or_else(|| {
        DataCollectionError::InvalidModelFormat(format!("tensor {name} has overflowing bounds"))
    })?;
    if end > float_data.len() {
        return Err(DataCollectionError::InvalidModelFormat(format!(
            "tensor {name} exceeds weights file bounds"
        )));
    }

    Ok(float_data[spec.offset_floats..end].to_vec())
}

fn masked_policy_from_logits(
    legal_actions: &[Action],
    policy_logits: &[f32],
    temperature: f32,
    turn: Player,
) -> Vec<(Action, f32)> {
    let temperature = temperature.max(0.05);
    let mut scaled_logits = Vec::with_capacity(legal_actions.len());
    let mut max_logit = f32::NEG_INFINITY;

    for action in legal_actions {
        let index = action.perspective_encoded_index(turn) as usize;
        let scaled = policy_logits[index] / temperature;
        max_logit = max_logit.max(scaled);
        scaled_logits.push(scaled);
    }

    let mut priors = Vec::with_capacity(legal_actions.len());
    let mut sum = 0.0;
    for scaled in &scaled_logits {
        let exp_value = (*scaled - max_logit).exp();
        priors.push(exp_value);
        sum += exp_value;
    }

    if !sum.is_finite() || sum <= 0.0 {
        let uniform = 1.0 / legal_actions.len() as f32;
        return legal_actions.iter().cloned().map(|action| (action, uniform)).collect();
    }

    legal_actions
        .iter()
        .cloned()
        .zip(priors.into_iter().map(|prior| prior / sum))
        .collect()
}

fn encode_state_features(state: &GameState, format: ModelFormat) -> Result<Vec<f32>, DataCollectionError> {
    match format {
        ModelFormat::V1 => encode_state_features_v1(state),
        ModelFormat::V2 => encode_state_features_v2(state),
    }
}

pub fn encode_teacher_feature_planes(state: &GameState) -> Result<Vec<f32>, DataCollectionError> {
    encode_teacher_feature_planes_with_version(state, TeacherFeatureLayoutVersion::LegacyV1)
}

pub fn encode_teacher_feature_planes_with_version(
    state: &GameState,
    version: TeacherFeatureLayoutVersion,
) -> Result<Vec<f32>, DataCollectionError> {
    let layout = TeacherFeaturePlaneLayout::with_version(version);
    let mut planes = vec![0.0; layout.input_dim()];
    let turn = state.turn();
    let opponent = turn.opponent();

    for index in 0..BOARD_CELLS {
        let position = Position::from_index(index)?;
        let perspective_index = to_perspective_index(index, turn);

        set_plane_value(
            &mut planes,
            20,
            perspective_index,
            center_score_normalized(position.row as usize, position.col as usize),
        );
        set_plane_value(
            &mut planes,
            21,
            perspective_index,
            edge_penalty_position(position),
        );

        if matches!(state.phase(), Phase::Placement)
            && state.is_legal_action(&Action::Place { at: position })
        {
            set_plane_value(&mut planes, 22, perspective_index, 1.0);
        }

        let Some(stack) = state.stack_at(position) else {
            continue;
        };

        let support = normalize_nonnegative(local_support_score(state, position, stack.owner), LOCAL_SUPPORT_SCALE);
        let attack = normalize_nonnegative(
            local_attack_score(state, position, stack.owner, stack.height),
            LOCAL_ATTACK_SCALE,
        );
        let vulnerability = normalize_nonnegative(
            local_vulnerability_score(state, position, stack.owner, stack.height),
            LOCAL_VULNERABILITY_SCALE,
        );

        if stack.owner == turn {
            set_teacher_height_bucket_plane(
                &mut planes,
                perspective_index,
                stack.height,
                true,
                version,
            );
            set_plane_value(&mut planes, 10, perspective_index, 1.0);
            set_plane_value(&mut planes, 12, perspective_index, support);
            set_plane_value(&mut planes, 14, perspective_index, attack);
            set_plane_value(&mut planes, 16, perspective_index, vulnerability);
        } else {
            set_teacher_height_bucket_plane(
                &mut planes,
                perspective_index,
                stack.height,
                false,
                version,
            );
            set_plane_value(&mut planes, 11, perspective_index, 1.0);
            set_plane_value(&mut planes, 13, perspective_index, support);
            set_plane_value(&mut planes, 15, perspective_index, attack);
            set_plane_value(&mut planes, 17, perspective_index, vulnerability);
        }

        if matches!(version, TeacherFeatureLayoutVersion::LegacyV1) {
            let cascade = normalize_nonnegative(
                local_cascade_potential(state, position, stack.owner),
                LOCAL_CASCADE_SCALE,
            );
            if stack.owner == turn {
                set_plane_value(&mut planes, 18, perspective_index, cascade);
            } else {
                set_plane_value(&mut planes, 19, perspective_index, cascade);
            }
        }
    }

    fill_broadcast_plane(
        &mut planes,
        23,
        matches!(state.phase(), Phase::Placement) as u8 as f32,
    );
    fill_broadcast_plane(
        &mut planes,
        24,
        matches!(state.phase(), Phase::Play) as u8 as f32,
    );
    fill_broadcast_plane(
        &mut planes,
        25,
        (state.repetition_count() as f32 / MAX_REPETITIONS).min(1.0),
    );
    if matches!(version, TeacherFeatureLayoutVersion::HeightBucketsV2)
        && matches!(state.phase(), Phase::Play)
    {
        populate_cascade_coverage_plane(&mut planes, 26, state, turn, turn)?;
        populate_cascade_coverage_plane(&mut planes, 27, state, opponent, turn)?;
    } else {
        fill_broadcast_plane(
            &mut planes,
            26,
            (state.play_turn() as f32 / MAX_PLAY_TURN).min(1.0),
        );
        fill_broadcast_plane(&mut planes, 27, (state.ply() as f32 / MAX_PLY).min(1.0));
    }
    fill_broadcast_plane(
        &mut planes,
        28,
        (state.piece_count(turn) as f32 / MAX_PIECES_PER_PLAYER).min(1.0),
    );
    fill_broadcast_plane(
        &mut planes,
        29,
        (state.piece_count(opponent) as f32 / MAX_PIECES_PER_PLAYER).min(1.0),
    );
    fill_broadcast_plane(
        &mut planes,
        30,
        (state.stack_count(turn) as f32 / MAX_STACKS_PER_PLAYER).min(1.0),
    );
    fill_broadcast_plane(
        &mut planes,
        31,
        (state.stack_count(opponent) as f32 / MAX_STACKS_PER_PLAYER).min(1.0),
    );

    Ok(planes)
}

fn set_teacher_height_bucket_plane(
    planes: &mut [f32],
    perspective_index: usize,
    height: u8,
    own_piece: bool,
    version: TeacherFeatureLayoutVersion,
) {
    if height == 0 {
        return;
    }

    if matches!(version, TeacherFeatureLayoutVersion::HeightBucketsV2) && height >= 6 {
        set_plane_value(planes, if own_piece { 18 } else { 19 }, perspective_index, 1.0);
        return;
    }

    let height_idx = (height.min(5) as usize).saturating_sub(1);
    let plane_index = if own_piece { height_idx } else { 5 + height_idx };
    set_plane_value(planes, plane_index, perspective_index, 1.0);
}

fn populate_cascade_coverage_plane(
    planes: &mut [f32],
    plane_index: usize,
    state: &GameState,
    player: Player,
    perspective_player: Player,
) -> Result<(), DataCollectionError> {
    for index in 0..BOARD_CELLS {
        let position = Position::from_index(index)?;
        let Some(stack) = state.stack_at(position) else {
            continue;
        };
        if stack.owner != player || stack.height < 2 {
            continue;
        }

        for direction in Direction::ALL {
            let mut current = position;
            for _ in 0..stack.height {
                let Some(next) = current.checked_step(direction) else {
                    break;
                };
                let perspective_index = to_perspective_index(next.index(), perspective_player);
                set_plane_value(planes, plane_index, perspective_index, 1.0);
                current = next;
            }
        }
    }

    Ok(())
}

fn encode_state_features_v1(state: &GameState) -> Result<Vec<f32>, DataCollectionError> {
    let layout = ModelFormat::V1.layout();
    let mut features = vec![0.0; layout.input_dim()];
    let turn = state.turn();
    let mut own_piece_count = 0.0;
    let mut opp_piece_count = 0.0;
    let mut own_stack_count = 0.0;
    let mut opp_stack_count = 0.0;

    for index in 0..BOARD_CELLS {
        let position = Position::from_index(index)?;
        let perspective_index = to_perspective_index(index, turn);
        let row = perspective_index / BOARD_SIDE as usize;
        let col = perspective_index % BOARD_SIDE as usize;
        let offset = perspective_index * layout.cell_feature_dim;

        features[offset + 4] = center_score_normalized(row, col);
        features[offset + 5] = edge_penalty(row, col);

        let Some(stack) = state.stack_at(position) else {
            continue;
        };

        let normalized_height = (stack.height as f32 / MAX_STACK_HEIGHT).min(1.0);
        if stack.owner == turn {
            features[offset] = normalized_height;
            features[offset + 2] = 1.0;
            own_piece_count += stack.height as f32;
            own_stack_count += 1.0;
        } else {
            features[offset + 1] = normalized_height;
            features[offset + 3] = 1.0;
            opp_piece_count += stack.height as f32;
            opp_stack_count += 1.0;
        }
    }

    populate_base_global_features(
        &mut features,
        BOARD_CELLS * layout.cell_feature_dim,
        state,
        own_piece_count,
        opp_piece_count,
        own_stack_count,
        opp_stack_count,
    );
    Ok(features)
}

fn encode_state_features_v2(state: &GameState) -> Result<Vec<f32>, DataCollectionError> {
    let layout = ModelFormat::V2.layout();
    let mut features = vec![0.0; layout.input_dim()];
    let turn = state.turn();
    let opponent = turn.opponent();
    let mut own_piece_count = 0.0;
    let mut opp_piece_count = 0.0;
    let mut own_stack_count = 0.0;
    let mut opp_stack_count = 0.0;

    for index in 0..BOARD_CELLS {
        let position = Position::from_index(index)?;
        let perspective_index = to_perspective_index(index, turn);
        let row = perspective_index / BOARD_SIDE as usize;
        let col = perspective_index % BOARD_SIDE as usize;
        let offset = perspective_index * layout.cell_feature_dim;

        features[offset + 4] = center_score_normalized(row, col);
        features[offset + 5] = edge_penalty(row, col);

        let Some(stack) = state.stack_at(position) else {
            continue;
        };

        let height = stack.height as f32;
        let normalized_height = (height / MAX_STACK_HEIGHT).min(1.0);
        let support = normalize_nonnegative(local_support_score(state, position, stack.owner), LOCAL_SUPPORT_SCALE);
        let attack = normalize_nonnegative(local_attack_score(state, position, stack.owner, stack.height), LOCAL_ATTACK_SCALE);
        let vulnerability = normalize_nonnegative(
            local_vulnerability_score(state, position, stack.owner, stack.height),
            LOCAL_VULNERABILITY_SCALE,
        );
        let cascade = normalize_nonnegative(local_cascade_potential(state, position, stack.owner), LOCAL_CASCADE_SCALE);

        if stack.owner == turn {
            features[offset] = normalized_height;
            features[offset + 2] = 1.0;
            features[offset + 6] = support;
            features[offset + 8] = attack;
            features[offset + 10] = vulnerability;
            features[offset + 12] = cascade;
            own_piece_count += height;
            own_stack_count += 1.0;
        } else {
            features[offset + 1] = normalized_height;
            features[offset + 3] = 1.0;
            features[offset + 7] = support;
            features[offset + 9] = attack;
            features[offset + 11] = vulnerability;
            features[offset + 13] = cascade;
            opp_piece_count += height;
            opp_stack_count += 1.0;
        }
    }

    let global_offset = BOARD_CELLS * layout.cell_feature_dim;
    populate_base_global_features(
        &mut features,
        global_offset,
        state,
        own_piece_count,
        opp_piece_count,
        own_stack_count,
        opp_stack_count,
    );
    features[global_offset + 10] = normalize_signed(attack_pressure(state, turn), GLOBAL_ATTACK_PRESSURE_SCALE);
    features[global_offset + 11] = normalize_signed(attack_pressure(state, opponent), GLOBAL_ATTACK_PRESSURE_SCALE);
    features[global_offset + 12] = normalize_nonnegative(vulnerable_mass(state, turn), GLOBAL_VULNERABLE_MASS_SCALE);
    features[global_offset + 13] = normalize_nonnegative(vulnerable_mass(state, opponent), GLOBAL_VULNERABLE_MASS_SCALE);
    features[global_offset + 14] = normalize_nonnegative(support_score(state, turn), GLOBAL_SUPPORT_SCALE);
    features[global_offset + 15] = normalize_nonnegative(support_score(state, opponent), GLOBAL_SUPPORT_SCALE);
    features[global_offset + 16] = normalize_nonnegative(center_mass(state, turn), GLOBAL_CENTER_MASS_SCALE);
    features[global_offset + 17] = normalize_nonnegative(center_mass(state, opponent), GLOBAL_CENTER_MASS_SCALE);
    features[global_offset + 18] = normalize_nonnegative(edge_exposure(state, turn), GLOBAL_EDGE_EXPOSURE_SCALE);
    features[global_offset + 19] = normalize_nonnegative(edge_exposure(state, opponent), GLOBAL_EDGE_EXPOSURE_SCALE);
    features[global_offset + 20] = normalize_signed(stack_profile(state, turn), GLOBAL_STACK_PROFILE_SCALE);
    features[global_offset + 21] = normalize_signed(stack_profile(state, opponent), GLOBAL_STACK_PROFILE_SCALE);
    features[global_offset + 22] = normalize_nonnegative(cascade_potential_total(state, turn), GLOBAL_CASCADE_POTENTIAL_SCALE);
    features[global_offset + 23] = normalize_nonnegative(cascade_potential_total(state, opponent), GLOBAL_CASCADE_POTENTIAL_SCALE);

    Ok(features)
}

fn populate_base_global_features(
    features: &mut [f32],
    global_offset: usize,
    state: &GameState,
    own_piece_count: f32,
    opp_piece_count: f32,
    own_stack_count: f32,
    opp_stack_count: f32,
) {
    features[global_offset] = matches!(state.phase(), Phase::Placement) as u8 as f32;
    features[global_offset + 1] = matches!(state.phase(), Phase::Play) as u8 as f32;
    features[global_offset + 2] = (state.repetition_count() as f32 / MAX_REPETITIONS).min(1.0);
    features[global_offset + 3] = (state.play_turn() as f32 / MAX_PLAY_TURN).min(1.0);
    features[global_offset + 4] = (state.ply() as f32 / MAX_PLY).min(1.0);
    features[global_offset + 5] = (own_piece_count / MAX_PIECES_PER_PLAYER).min(1.0);
    features[global_offset + 6] = (opp_piece_count / MAX_PIECES_PER_PLAYER).min(1.0);
    features[global_offset + 7] = (own_stack_count / MAX_STACKS_PER_PLAYER).min(1.0);
    features[global_offset + 8] = (opp_stack_count / MAX_STACKS_PER_PLAYER).min(1.0);
    features[global_offset + 9] = ((own_piece_count + opp_piece_count) / (MAX_PIECES_PER_PLAYER * 2.0)).min(1.0);
}

fn to_perspective_index(index: usize, turn: Player) -> usize {
    let mut row = index / BOARD_SIDE as usize;
    let mut col = index % BOARD_SIDE as usize;
    if matches!(turn, Player::Blue) {
        row = BOARD_SIDE as usize - 1 - row;
        col = BOARD_SIDE as usize - 1 - col;
    }
    row * BOARD_SIDE as usize + col
}

fn set_plane_value(values: &mut [f32], plane_index: usize, cell_index: usize, value: f32) {
    values[plane_index * BOARD_CELLS + cell_index] = value;
}

fn fill_broadcast_plane(values: &mut [f32], plane_index: usize, value: f32) {
    let start = plane_index * BOARD_CELLS;
    let end = start + BOARD_CELLS;
    values[start..end].fill(value);
}

fn center_score_normalized(row: usize, col: usize) -> f32 {
    let row_distance = (row as f32 - 3.5).abs();
    let col_distance = (col as f32 - 3.5).abs();
    ((4.0 - 0.55 * (row_distance + col_distance)) / 4.0).max(0.0)
}

fn center_score_raw(position: Position) -> f32 {
    let row_distance = (position.row as f32 - 3.5).abs();
    let col_distance = (position.col as f32 - 3.5).abs();
    (4.0 - 0.55 * (row_distance + col_distance)).max(0.0)
}

fn edge_penalty(row: usize, col: usize) -> f32 {
    let distance = row.min(BOARD_SIDE as usize - 1 - row).min(col.min(BOARD_SIDE as usize - 1 - col));
    match distance {
        0 => 1.0,
        1 => 0.45,
        2 => 0.12,
        _ => 0.0,
    }
}

fn edge_penalty_position(position: Position) -> f32 {
    edge_penalty(position.row as usize, position.col as usize)
}

fn normalize_nonnegative(value: f32, scale: f32) -> f32 {
    if scale <= 0.0 {
        0.0
    } else {
        (value / scale).clamp(0.0, 1.0)
    }
}

fn normalize_signed(value: f32, scale: f32) -> f32 {
    if scale <= 0.0 {
        0.0
    } else {
        (value / scale).clamp(-1.0, 1.0)
    }
}

fn local_support_score(state: &GameState, position: Position, player: Player) -> f32 {
    Direction::ALL
        .into_iter()
        .filter_map(|direction| position.checked_step(direction))
        .filter_map(|neighbor| state.stack_at(neighbor))
        .filter(|stack| stack.owner == player)
        .map(|stack| 0.35 + stack.height as f32 * 0.08)
        .sum()
}

fn local_attack_score(state: &GameState, position: Position, player: Player, height: u8) -> f32 {
    let mut score = 0.0;
    for direction in Direction::ALL {
        let Some(neighbor) = position.checked_step(direction) else {
            continue;
        };
        let Some(enemy_stack) = state.stack_at(neighbor) else {
            continue;
        };
        if enemy_stack.owner == player {
            continue;
        }
        if height >= enemy_stack.height {
            score += 1.0 + (height - enemy_stack.height) as f32 * 0.2;
        }
    }
    score
}

fn local_vulnerability_score(state: &GameState, position: Position, player: Player, height: u8) -> f32 {
    let mut score = 0.0;
    for direction in Direction::ALL {
        let Some(neighbor) = position.checked_step(direction) else {
            continue;
        };
        let Some(enemy_stack) = state.stack_at(neighbor) else {
            continue;
        };
        if enemy_stack.owner == player {
            continue;
        }
        if enemy_stack.height >= height {
            score += 1.0 + (enemy_stack.height - height) as f32 * 0.2;
        }
    }
    score
}

fn local_cascade_potential(state: &GameState, position: Position, player: Player) -> f32 {
    Direction::ALL
        .into_iter()
        .map(|direction| cascade_lane_direction_score(state, position, direction, player))
        .fold(0.0, f32::max)
}

fn cascade_lane_direction_score(
    state: &GameState,
    from: Position,
    direction: Direction,
    player: Player,
) -> f32 {
    let mut score = 0.0;
    let mut current = from;

    for _ in 0..3 {
        let Some(next) = current.checked_step(direction) else {
            break;
        };

        score += center_score_raw(next) * 0.04;
        if let Some(stack) = state.stack_at(next) {
            if stack.owner == player.opponent() {
                score += 0.12 + stack.height as f32 * 0.06;
                if edge_penalty_position(next) > 0.0 {
                    score += 0.08 * stack.height as f32;
                }
            } else {
                score += 0.03 * stack.height as f32;
            }
        }

        current = next;
    }

    score
}

fn attack_pressure(state: &GameState, player: Player) -> f32 {
    let mut score = 0.0;

    for index in 0..BOARD_CELLS {
        let position = Position::from_index(index).expect("board index is valid");
        let Some(stack) = state.stack_at(position) else {
            continue;
        };
        if stack.owner != player {
            continue;
        }

        for direction in Direction::ALL {
            let Some(target) = position.checked_step(direction) else {
                continue;
            };
            let Some(target_stack) = state.stack_at(target) else {
                continue;
            };
            if target_stack.owner == player {
                continue;
            }
            if stack.height >= target_stack.height {
                score += 1.0 + (stack.height - target_stack.height) as f32 * 0.2;
            }
            if target_stack.height >= stack.height {
                score -= 0.45 + (target_stack.height - stack.height) as f32 * 0.1;
            }
        }
    }

    score
}

fn vulnerable_mass(state: &GameState, player: Player) -> f32 {
    let mut total = 0.0;

    for index in 0..BOARD_CELLS {
        let position = Position::from_index(index).expect("board index is valid");
        let Some(stack) = state.stack_at(position) else {
            continue;
        };
        if stack.owner != player {
            continue;
        }

        let attackers = local_vulnerability_score(state, position, player, stack.height);
        if attackers > 0.0 {
            total += stack.height as f32 * attackers;
        }
    }

    total
}

fn support_score(state: &GameState, player: Player) -> f32 {
    let mut total = 0.0;

    for index in 0..BOARD_CELLS {
        let position = Position::from_index(index).expect("board index is valid");
        let Some(stack) = state.stack_at(position) else {
            continue;
        };
        if stack.owner != player {
            continue;
        }

        total += local_support_score(state, position, player) + stack.height as f32 * 0.04;
    }

    total * 0.5
}

fn stack_profile(state: &GameState, player: Player) -> f32 {
    let mut total = 0.0;

    for index in 0..BOARD_CELLS {
        let position = Position::from_index(index).expect("board index is valid");
        let Some(stack) = state.stack_at(position) else {
            continue;
        };
        if stack.owner != player {
            continue;
        }

        total += match stack.height {
            1 => -0.30,
            2 => 0.10,
            3 => 0.24,
            4 => 0.36,
            5 => 0.42,
            _ => 0.42 + (stack.height as f32 - 5.0) * 0.04,
        };
    }

    total
}

fn center_mass(state: &GameState, player: Player) -> f32 {
    let mut total = 0.0;

    for index in 0..BOARD_CELLS {
        let position = Position::from_index(index).expect("board index is valid");
        let Some(stack) = state.stack_at(position) else {
            continue;
        };
        if stack.owner == player {
            total += center_score_normalized(position.row as usize, position.col as usize) * stack.height as f32;
        }
    }

    total
}

fn edge_exposure(state: &GameState, player: Player) -> f32 {
    let mut total = 0.0;

    for index in 0..BOARD_CELLS {
        let position = Position::from_index(index).expect("board index is valid");
        let Some(stack) = state.stack_at(position) else {
            continue;
        };
        if stack.owner == player {
            total += edge_penalty_position(position) * stack.height as f32;
        }
    }

    total
}

fn cascade_potential_total(state: &GameState, player: Player) -> f32 {
    let mut total = 0.0;

    for index in 0..BOARD_CELLS {
        let position = Position::from_index(index).expect("board index is valid");
        let Some(stack) = state.stack_at(position) else {
            continue;
        };
        if stack.owner == player {
            total += local_cascade_potential(state, position, player);
        }
    }

    total
}

fn layer_norm_in_place(
    values: &mut [f32],
    weight: &[f32],
    bias: &[f32],
    eps: f32,
) -> Result<(), DataCollectionError> {
    if values.len() != weight.len() || values.len() != bias.len() {
        return Err(DataCollectionError::InvalidModelFormat(
            "layer norm parameter shape mismatch".to_string(),
        ));
    }

    let count = values.len() as f32;
    let mean = values.iter().copied().sum::<f32>() / count;
    let variance = values
        .iter()
        .map(|value| {
            let centered = *value - mean;
            centered * centered
        })
        .sum::<f32>()
        / count;
    let inv_std = 1.0 / (variance + eps).sqrt();

    for index in 0..values.len() {
        values[index] = ((values[index] - mean) * inv_std) * weight[index] + bias[index];
    }

    Ok(())
}

fn gelu_in_place(values: &mut [f32]) {
    for value in values {
        let x = *value;
        *value = 0.5 * x * (1.0 + erff(x / SQRT_2));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BoardState, GameStateSnapshot};

    #[test]
    fn masked_policy_normalizes_over_legal_actions() {
        let legal_actions = vec![
            Action::from_encoded_index(1).unwrap(),
            Action::from_encoded_index(27).unwrap(),
            Action::from_encoded_index(65).unwrap(),
        ];
        let mut logits = vec![0.0; TOTAL_ACTIONS];
        logits[1] = 0.5;
        logits[27] = 1.0;
        logits[65] = -1.0;

        let priors = masked_policy_from_logits(&legal_actions, &logits, 1.0, Player::Red);
        let total: f32 = priors.iter().map(|(_, prior)| *prior).sum();

        assert!((total - 1.0).abs() < 1e-6);
        assert!(priors[1].1 > priors[0].1);
        assert!(priors[0].1 > priors[2].1);
    }

    #[test]
    fn teacher_feature_planes_encode_initial_placement_mask() {
        let features =
            encode_teacher_feature_planes_with_version(
                &GameState::new(),
                TeacherFeatureLayoutVersion::LegacyV1,
            )
            .unwrap();
        let layout = TeacherFeaturePlaneLayout::new();
        let mask_plane = &features[22 * BOARD_CELLS..23 * BOARD_CELLS];
        let placement_phase_plane = &features[23 * BOARD_CELLS..24 * BOARD_CELLS];

        assert_eq!(features.len(), layout.input_dim());
        assert_eq!(mask_plane.iter().filter(|value| **value > 0.5).count(), BOARD_CELLS);
        assert!(placement_phase_plane.iter().all(|value| (*value - 1.0).abs() < 1e-6));
    }

    #[test]
    fn teacher_feature_planes_respect_non_adjacent_placement_rule() {
        let mut state = GameState::new();
        state
            .apply_action(Action::Place {
                at: Position::new(3, 3).unwrap(),
            })
            .unwrap();

        let features = encode_teacher_feature_planes_with_version(
            &state,
            TeacherFeatureLayoutVersion::LegacyV1,
        )
        .unwrap();
        let mask_plane = &features[22 * BOARD_CELLS..23 * BOARD_CELLS];

        assert_eq!(mask_plane.iter().filter(|value| **value > 0.5).count(), 59);
    }

    #[test]
    fn teacher_feature_planes_height_buckets_v2_use_planes_18_and_19_for_six_plus_stacks() {
        let mut state = GameState::new();
        state
            .set_stack(Position::new(0, 0).unwrap(), Player::Red, 6)
            .unwrap();
        state
            .set_stack(Position::new(0, 1).unwrap(), Player::Blue, 7)
            .unwrap();

        let legacy = encode_teacher_feature_planes_with_version(
            &state,
            TeacherFeatureLayoutVersion::LegacyV1,
        )
        .unwrap();
        let v2 = encode_teacher_feature_planes_with_version(
            &state,
            TeacherFeatureLayoutVersion::HeightBucketsV2,
        )
        .unwrap();

        assert!((legacy[4 * BOARD_CELLS] - 1.0).abs() < 1e-6);
        assert!(legacy[18 * BOARD_CELLS] > 0.0);
        assert_eq!(v2[4 * BOARD_CELLS], 0.0);
        assert!((v2[18 * BOARD_CELLS] - 1.0).abs() < 1e-6);

        assert!((legacy[9 * BOARD_CELLS + 1] - 1.0).abs() < 1e-6);
        assert!(legacy[19 * BOARD_CELLS + 1] > 0.0);
        assert_eq!(v2[9 * BOARD_CELLS + 1], 0.0);
        assert!((v2[19 * BOARD_CELLS + 1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn teacher_feature_planes_height_buckets_v2_flip_cascade_coverage_with_turn() {
        let mut board = BoardState::empty();
        board
            .set_stack(Position::new(6, 6).unwrap(), Player::Blue, 2)
            .unwrap();
        board
            .set_stack(Position::new(1, 1).unwrap(), Player::Red, 2)
            .unwrap();

        let snapshot = GameStateSnapshot::new(Phase::Play, Player::Blue, board).unwrap();
        let state = GameState::from_snapshot(snapshot).unwrap();
        let features = encode_teacher_feature_planes_with_version(
            &state,
            TeacherFeatureLayoutVersion::HeightBucketsV2,
        )
        .unwrap();

        let own_target = Position::new(5, 6).unwrap();
        let own_flipped = to_perspective_index(own_target.index(), Player::Blue);
        let opp_target = Position::new(2, 1).unwrap();
        let opp_flipped = to_perspective_index(opp_target.index(), Player::Blue);

        assert_eq!(features[26 * BOARD_CELLS + own_flipped], 1.0);
        assert_eq!(features[27 * BOARD_CELLS + opp_flipped], 1.0);
        assert_eq!(features[26 * BOARD_CELLS + own_target.index()], 0.0);
        assert_eq!(features[27 * BOARD_CELLS + opp_target.index()], 0.0);
    }

    #[test]
    fn teacher_onnx_manifest_defaults_to_cuda() {
        let manifest: TeacherOnnxManifest = serde_json::from_value(serde_json::json!({
            "format": "cascade_teacher_onnx_v1",
            "onnx_file": "teacher_resnet.onnx",
            "feature_planes": TEACHER_FEATURE_PLANES,
            "board_side": BOARD_SIDE as usize,
            "action_dim": TOTAL_ACTIONS,
        }))
        .unwrap();

        assert_eq!(manifest.execution_provider, TeacherExecutionProvider::Cuda);
        assert_eq!(manifest.feature_layout_version, TeacherFeatureLayoutVersion::LegacyV1);
        assert_eq!(manifest.input_name, DEFAULT_TEACHER_INPUT_NAME);
        assert_eq!(manifest.policy_output_name, DEFAULT_TEACHER_POLICY_OUTPUT_NAME);
        assert_eq!(manifest.value_output_name, DEFAULT_TEACHER_VALUE_OUTPUT_NAME);
        assert!(!manifest.allow_cpu_fallback);
    }
}
