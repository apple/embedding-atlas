/// Progress callback: receives (progress, stage) where progress is in [0.0, 1.0].
pub type ProgressCallback = Box<dyn FnMut(f32, &str)>;

/// Unified logger that handles verbose text output and optional progress callbacks.
///
/// Always instantiated (not `Option`). The progress callback inside is optional.
/// Progress is computed from stage definitions: each stage has an estimated fraction
/// of total time, and `stage_progress()` maps within-stage progress to overall [0, 1].
pub struct Logger {
    verbose: bool,
    pub callback: Option<ProgressCallback>,
    stages: Vec<(&'static str, f32)>,
    cumulative: Vec<f32>,
    current_stage: Option<usize>,
}

impl Logger {
    /// Create a new logger with the given verbosity, optional progress callback,
    /// and stage definitions (pairs of stage name and estimated time fraction).
    pub fn new(
        verbose: bool,
        callback: Option<ProgressCallback>,
        stages: &[(&'static str, f32)],
    ) -> Self {
        let mut cumulative = Vec::with_capacity(stages.len());
        let mut sum = 0.0f32;
        for &(_, frac) in stages {
            cumulative.push(sum);
            sum += frac;
        }
        Self {
            verbose,
            callback,
            stages: stages.to_vec(),
            cumulative,
            current_stage: None,
        }
    }

    /// Print a message to stderr if verbose is enabled.
    pub fn log(&self, msg: &str) {
        if self.verbose {
            eprintln!("{}", msg);
        }
    }

    /// Push (enter) a stage. Reports progress at stage start.
    pub fn push_stage(&mut self, stage: &str) {
        let idx = self.stages.iter().position(|(s, _)| *s == stage);
        self.current_stage = idx;
        if let (Some(idx), Some(ref mut cb)) = (self.current_stage, &mut self.callback) {
            cb(self.cumulative[idx], stage);
        }
    }

    /// Push (enter) a stage with a custom verbose message.
    /// Reports progress at stage start and logs the message.
    pub fn push_stage_with_message(&mut self, stage: &str, msg: &str) {
        self.push_stage(stage);
        self.log(msg);
    }

    /// Report sub-progress within the current stage.
    ///
    /// `progress` is 0.0 to 1.0 within this stage.
    /// `status` optionally overrides the stage name in the callback.
    pub fn stage_progress(&mut self, progress: f64, status: Option<&str>) {
        if let Some(idx) = self.current_stage {
            let (name, frac) = self.stages[idx];
            let overall = self.cumulative[idx] + frac * progress as f32;
            if let Some(ref mut cb) = self.callback {
                cb(overall.min(1.0), status.unwrap_or(name));
            }
        }
    }

    /// Pop (exit) the current stage. Reports progress at stage end.
    pub fn pop_stage(&mut self) {
        if let Some(idx) = self.current_stage {
            if let Some(ref mut cb) = self.callback {
                let end = self.cumulative[idx] + self.stages[idx].1;
                let name = self.stages[idx].0;
                cb(end.min(1.0), name);
            }
        }
        self.current_stage = None;
    }
}
