use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

pub(crate) struct PeriodicReporter {
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl PeriodicReporter {
    pub(crate) fn spawn<F>(interval_millis: u64, render: F) -> Self
    where
        F: Fn() -> Option<String> + Send + Sync + 'static,
    {
        let stop = Arc::new(AtomicBool::new(false));
        let stop_for_thread = Arc::clone(&stop);
        let render = Arc::new(render);
        let render_for_thread = Arc::clone(&render);
        let interval = Duration::from_millis(interval_millis.max(250));

        let handle = thread::Builder::new()
            .name("cascade-progress".to_string())
            .spawn(move || {
                while !stop_for_thread.load(Ordering::Acquire) {
                    thread::sleep(interval);
                    if stop_for_thread.load(Ordering::Acquire) {
                        break;
                    }
                    if let Some(line) = render_for_thread() {
                        eprintln!("{line}");
                    }
                }
            })
            .ok();

        Self { stop, handle }
    }

    pub(crate) fn stop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for PeriodicReporter {
    fn drop(&mut self) {
        self.stop();
    }
}
