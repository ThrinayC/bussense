/**
 * BusSense Web Audio API Synthetic Acoustic Warning Engine
 * Zero external MP3 assets, 100% offline synthesis.
 */
let audioCtx = null;
let soundEnabled = localStorage.getItem('bussense_audio_alerts') !== 'false';

function getAudioContext() {
  if (!audioCtx) {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (AudioContextClass) audioCtx = new AudioContextClass();
  }
  if (audioCtx && audioCtx.state === 'suspended') {
    audioCtx.resume();
  }
  return audioCtx;
}

export function isAudioEnabled() {
  return soundEnabled;
}

export function setAudioEnabled(enabled) {
  soundEnabled = Boolean(enabled);
  localStorage.setItem('bussense_audio_alerts', soundEnabled ? 'true' : 'false');
  if (soundEnabled) playSuccessTone();
}

export function toggleAudio() {
  setAudioEnabled(!soundEnabled);
  return soundEnabled;
}

/**
 * Play a double-harmonic warning tone for critical perception alerts
 */
export function playCriticalAlarm() {
  if (!soundEnabled) return;
  try {
    const ctx = getAudioContext();
    if (!ctx) return;
    const now = ctx.currentTime;

    // Harmonic 1: 880Hz (A5)
    const osc1 = ctx.createOscillator();
    const gain1 = ctx.createGain();
    osc1.type = 'sine';
    osc1.frequency.setValueAtTime(880, now);
    osc1.frequency.exponentialRampToValueAtTime(440, now + 0.28);
    gain1.gain.setValueAtTime(0.2, now);
    gain1.gain.exponentialRampToValueAtTime(0.001, now + 0.28);
    osc1.connect(gain1);
    gain1.connect(ctx.destination);
    osc1.start(now);
    osc1.stop(now + 0.3);

    // Harmonic 2: 1760Hz (A6 ping)
    const osc2 = ctx.createOscillator();
    const gain2 = ctx.createGain();
    osc2.type = 'triangle';
    osc2.frequency.setValueAtTime(1760, now + 0.12);
    osc2.frequency.exponentialRampToValueAtTime(880, now + 0.35);
    gain2.gain.setValueAtTime(0.15, now + 0.12);
    gain2.gain.exponentialRampToValueAtTime(0.001, now + 0.35);
    osc2.connect(gain2);
    gain2.connect(ctx.destination);
    osc2.start(now + 0.12);
    osc2.stop(now + 0.36);
  } catch (e) {
    console.warn('Audio synthesis failed:', e);
  }
}

export function playSuccessTone() {
  try {
    const ctx = getAudioContext();
    if (!ctx) return;
    const now = ctx.currentTime;
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(587.33, now); // D5
    osc.frequency.setValueAtTime(880, now + 0.08); // A5
    gain.gain.setValueAtTime(0.08, now);
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.2);
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start(now);
    osc.stop(now + 0.22);
  } catch (e) {}
}
