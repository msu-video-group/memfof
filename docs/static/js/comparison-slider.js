const sliders = [];

function createComparisonSliderForSection(section_id, fps) {
    const FRAME_DURATION_MS = 1 / fps;
    const SYNC_CHECK_INTERVAL_MS = 200;
    const MAX_DESYNC_TIME_MS = 3 * FRAME_DURATION_MS;

    const parent = document.getElementById(section_id);
    if (!parent) {
        console.error(`No section with id="${section_id}"`);
        return;
    }

    const slider = parent.querySelector('img-comparison-slider');
    if (!slider) {
        console.error('No <img-comparison-slider> inside', parent);
        return;
    }

    const slot_first = slider.querySelector('video[slot="first"]');
    const slot_second = slider.querySelector('video[slot="second"]');

    if (!slot_first || !slot_second) {
        console.error('Videos with slots "first" and "second" not found');
        return;
    }

    const waitForMetadata = (video) => {
        return new Promise(resolve => {
            if (video.readyState >= 1) {
                resolve();
            } else {
                video.addEventListener('loadedmetadata', () => resolve(), { once: true });
            }
        });
    };

    Promise.all([waitForMetadata(slot_first), waitForMetadata(slot_second)]).then(() => {
        slot_second.currentTime = slot_first.currentTime;

        function syncCurrentTime() {
            slot_second.currentTime = slot_first.currentTime;
        }

        function syncPlay() {
            slot_first.play();
            slot_second.play();
        }

        function syncPauseAtPrevFrame() {
            slot_first.pause();
            slot_second.pause();
            slot_first.currentTime = Math.max(0, slot_first.currentTime - FRAME_DURATION_MS);
            syncCurrentTime();
        }

        function syncPauseAtNextFrame() {
            slot_first.pause();
            slot_second.pause();
            slot_first.currentTime = Math.min(slot_first.duration, slot_first.currentTime + FRAME_DURATION_MS);
            syncCurrentTime();
        }

        function togglePlayPause() {
            if (slot_first.paused) {
                syncPlay();
            } else {
                syncPauseAtNextFrame();
            }
        }

        slider.addEventListener('click', togglePlayPause);

        slot_first.addEventListener('play', syncPlay);
        slot_first.addEventListener('pause', syncPauseAtNextFrame);

        function getWrappedDelta(t1, t2, duration) {
            const rawDelta = Math.abs(t1 - t2);
            return Math.min(rawDelta, duration - rawDelta);
        }

        slider.dataset.syncIntervalId = setInterval(async () => {
            const delta = getWrappedDelta(slot_first.currentTime, slot_second.currentTime, slot_first.duration);
            if (delta > MAX_DESYNC_TIME_MS) {
                console.log("Detected video desync of " + delta + " ms");
                syncPauseAtNextFrame();
                await new Promise(resolve => setTimeout(resolve, 200));
                syncPlay();
            }
        }, SYNC_CHECK_INTERVAL_MS);

        sliders.push({
            togglePlayPause,
            syncPauseAtNextFrame,
            syncPauseAtPrevFrame,
        });
    }).catch(err => {
        console.error('Error loading video', err);
    });
}

document.addEventListener('DOMContentLoaded', () => {
    createComparisonSliderForSection('spring-videos', 30);
    createComparisonSliderForSection('sintel-videos', 24);
    createComparisonSliderForSection('kitti-videos', 2);

    document.querySelectorAll('img-comparison-slider').forEach(slider => {
        slider.addEventListener('click', (e) => {
            slider.blur?.();
        });
    });
});

document.addEventListener('keydown', (e) => {
    if (e.code === 'Space') {
        e.preventDefault();
        sliders.forEach(s => s.togglePlayPause());
    } else if (e.code === 'ArrowRight') {
        e.preventDefault();
        sliders.forEach(s => s.syncPauseAtNextFrame());
    } else if (e.code === 'ArrowLeft') {
        e.preventDefault();
        sliders.forEach(s => s.syncPauseAtPrevFrame());
    }
});

window.addEventListener('beforeunload', () => {
    sliders.forEach(s => {
        const id = parseInt(s.slider?.dataset.syncIntervalId);
        if (!isNaN(id)) {
            clearInterval(id);
        }
    });
});
