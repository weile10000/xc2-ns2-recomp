import { api, esc, icon, refreshIcons, toast, errText } from '../util.js';
import { t } from '../i18n.js';
import { statusText, stepTitle } from '../rusttext.js';

const hex = (c) => '0x' + (c >>> 0).toString(16).toUpperCase().padStart(8, '0');

function voicePackLabel(stem) {
  const s = String(stem).trim().toLowerCase();
  if (s === 'zh_hans' || s === 'zh_cn') return t('settings.voiceLang.zhHans');
  if (s === 'zh_hant' || s === 'zh_tw') return t('settings.voiceLang.zhHant');
  if (s === 'en') return t('settings.voiceLang.en');
  if (s === 'ja') return t('settings.voiceLang.ja');
  return stem;
}

function statusChip(s) {
  if (s.state === 'running') return `<span class="chip warn">${icon('loader-circle', 'spin')}${esc(t('play.running'))}</span>`;
  if (s.state === 'exited') {
    const code = s.code === null || s.code === undefined ? t('play.noExitCode') : t('play.codeWithHex', { code: s.code, hex: hex(s.code) });
    return s.crashed
      ? `<span class="chip danger">${icon('circle-alert')}${esc(t('play.crashed', { code }))}</span>`
      : `<span class="chip ok">${icon('circle-check')}${esc(t('play.exited', { code }))}</span>`;
  }
  return `<span class="chip">${esc(t('play.ready'))}</span>`;
}

/* The line a run ends with comes from play.rs in English; rusttext.js translates it. */
const statusMsg = (s) => statusText(s);

function reportHtml(s) {
  if (s.state !== 'exited') return '';
  const files = s.files.filter((f) => s.crashed || f.name === 'fefw.log');
  return `
    <div class="card">
      <h2>${icon(s.crashed ? 'circle-alert' : 'file-text')}${esc(t(s.crashed ? 'play.reportCrashed' : 'play.reportClean'))}</h2>
      <p class="dim">${esc(statusMsg(s))}</p>
      ${files.length ? `<div class="row wrap" style="margin-top: var(--space-4)">${files
        .map((f) => `<button class="btn" data-open="${esc(f.name)}">${icon('file-text')}${esc(f.name)}${f.fresh && f.name !== 'fefw.log' ? esc(t('play.fileNew')) : ''}</button>`)
        .join('')}</div>` : ''}
      ${s.crashed ? `<p class="dim" style="margin-top: var(--space-4)">${esc(t('play.recordHint'))}</p>
      <div class="row wrap"><button class="btn primary" id="record">${icon('file-text')}${esc(t('play.recordReport'))}</button></div>` : ''}
    </div>`;
}

export async function render(el, ctx) {
  const [cfg, ov] = await Promise.all([api('get_config'), api('get_overview')]);
  const setup = ov.repo_ok ? await api('setup_status') : null;
  const report = cfg.dump_dir && ov.repo_ok ? await api('validate_dump', { path: cfg.dump_dir }) : null;
  const chosen = report && report.chosen;
  const modsSnap = ov.repo_ok ? await api('list_mods').catch(() => null) : null;
  let voiceStatus = ov.repo_ok ? await api('voice_subtitles_status').catch(() => null) : null;
  let status = await api('game_status');
  // Linux: fefw.exe as a non-Steam game (steam.rs)
  let steam = await api('steam_status').catch(() => ({ available: false }));
  let addingSteam = false;

  function draw() {
    const running = status.state === 'running';
    const problems = [];
    let upgradeNeeded = false;
    if (!ov.repo_ok) problems.push(t('play.problemRepo'));
    else if (!cfg.dump_dir) problems.push(t('play.problemNoDump'));
    else if (!chosen) problems.push(t('play.problemBadDump', { path: cfg.dump_dir, errors: report.errors.join(' ') }));
    else if (setup && setup.progress.state === 'running') problems.push(t('play.problemSetupRunning'));
    else if (setup && !setup.complete && setup.upgrade_needed) upgradeNeeded = true;
    else if (!setup || !setup.complete) problems.push(t('play.problemSetupUnfinished'));
    const ready = problems.length === 0 && !upgradeNeeded;
    const left = setup ? setup.steps.filter((x) => x.state !== 'done').map((x) => stepTitle(x)) : [];
    const upgradeReason = (setup && setup.upgrade_reason) || '';
    const upgradeBanner = upgradeNeeded
      ? `<div class="banner warn">${icon('circle-arrow-up')}<div>${esc(
          upgradeReason ? t('play.upgradeBannerWithReason', { reason: upgradeReason }) : t('play.upgradeBanner'),
        )}</div></div>`
      : '';

    el.innerHTML = `
      <div class="page">
        <div class="hero">
          <h1>${esc(t('play.title'))}</h1>
          <p class="dim lead">${esc(t('play.intro'))}</p>
          <p class="faint note">${esc(t('play.footer'))}</p>
        </div>
        ${problems.map((p) => `<div class="banner danger">${icon('circle-x')}<div>${esc(p)}</div></div>`).join('')}
        ${problems.length && ov.repo_ok ? `<div><button class="btn primary" id="to-setup">${icon(chosen ? 'hammer' : 'folder-open')}${esc(t(chosen ? 'play.openSetup' : 'play.chooseFiles'))}</button></div>` : ''}
        ${upgradeBanner}

        <div class="row wrap">
          ${
            upgradeNeeded && !running
              ? `<button class="btn primary big" id="finish-upgrade">${icon('hammer')}${esc(t('play.finishUpgrade'))}</button>
                 <span class="chip warn">${icon('circle-arrow-up')}${esc(t('play.upgradePending'))}</span>`
              : `<button class="btn primary big" id="play" ${ready && !running ? '' : 'disabled'}>${icon('play')}${esc(t(running ? 'play.running' : 'play.title'))}</button>
                 ${statusChip(status)}`
          }
        </div>
        <div class="banner love">${icon('heart')}<div>${esc(t('play.supportOfficial'))}</div></div>
        ${reportHtml(status)}

        <div class="card">
          <h2>${icon('hard-drive')}${esc(t('play.install'))}</h2>
          <dl class="kv">
            <dt>${esc(t('play.gameFiles'))}</dt>
            <dd>${chosen ? `${esc(chosen.kind === 'update' ? t('common.update', { version: chosen.version }) : chosen.kind === 'base' ? t('common.baseGame') : t('common.folder'))}<div class="path">${esc(chosen.path)}</div>` : `<span class="faint">${esc(t('common.notSet'))}</span>`}</dd>
            <dt>${esc(t('play.dlc'))}</dt>
            <dd>${report && report.dlc && report.dlc.length ? `${esc(t(report.dlc.length === 1 ? 'play.dlcFound' : 'play.dlcFoundMany', { count: report.dlc.length }))}<div class="path">${esc(report.dlc_dir)}</div>` : `<span class="faint">${esc(t('play.dlcNone'))}</span>`}</dd>
            <dt>${esc(t('play.setup'))}</dt>
            <dd>${setup && setup.complete ? esc(t('common.complete')) : `<span class="faint">${esc(upgradeNeeded ? t('common.upgradeNeeded', { list: left.join(', ') }) : left.length ? t('common.todo', { list: left.join(', ') }) : t('common.notStarted'))}</span>`}</dd>
            <dt>${esc(t('play.exe'))}</dt>
            <dd>${ov.exe_exists ? esc(t('common.built')) : `<span class="faint">${esc(t('common.notBuiltYet'))}</span>`}<div class="path">${esc(ov.exe)}</div></dd>
            <dt>${esc(t('play.runFolder'))}</dt>
            <dd><span class="faint">${esc(t('play.runFolderContents'))}</span><div class="path">${esc(ov.game_dir)}</div></dd>
            <dt>${esc(t('play.mods'))}</dt>
            <dd>${(() => {
              if (!modsSnap || !modsSnap.mods.length) return `<span class="faint">${esc(t('play.modsHint'))}</span>`;
              const enabled = modsSnap.mods.filter((m) => m.enabled).length;
              const disabled = modsSnap.mods.length - enabled;
              const base = t('mods.summaryChip', { enabled, disabled });
              const pending = modsSnap.mods.filter((m) => m.pending_rebuild).length;
              const text = pending ? `${base} · ${t('play.modsPendingRebuild', { count: pending })}` : base;
              return `${esc(text)}<div class="path">${esc(modsSnap.mods_dir)}</div>`;
            })()}</dd>
          </dl>
          <div class="row wrap" style="margin-top: var(--space-4)">
            <button class="btn" id="open-dir" ${ov.repo_ok ? '' : 'disabled'}>${icon('folder-open')}${esc(t('play.openRunFolder'))}</button>
            <button class="btn" id="manage-mods" ${ov.repo_ok ? '' : 'disabled'}>${icon('package')}${esc(t('play.manageMods'))}</button>
            <button class="btn" id="open-mods" ${ov.repo_ok ? '' : 'disabled'}>${icon('folder-open')}${esc(t('play.openModsFolder'))}</button>
            <button class="btn" id="import-saves" ${ov.repo_ok ? '' : 'disabled'} title="${esc(t('play.importSavesHint'))}">${icon('folder-open')}${esc(t('play.importSaves'))}</button>
            ${steam.available ? `<button class="btn" id="add-steam" title="${esc(t('play.addToSteamHint'))}" ${ov.exe_exists && !addingSteam ? '' : 'disabled'}>${icon(addingSteam ? 'loader-circle' : 'gamepad-2', addingSteam ? 'spin' : '')}${esc(t(steam.added ? 'play.updateSteam' : 'play.addToSteam'))}</button>` : ''}
          </div>
        </div>

        <div class="card">
          <h2>${icon('sparkles')}${esc(t('play.features'))}</h2>
          <dl class="kv">
            <dt>${esc(t('play.featureNav'))}</dt>
            <dd>${esc(t('play.featureNavDesc'))}</dd>
            <dt>${esc(t('play.featureVoice'))}</dt>
            <dd>
              <div>${esc(t('play.featureVoiceDesc'))}</div>
              <div class="faint" style="margin-top: var(--space-1)">${
                voiceStatus && voiceStatus.packs && voiceStatus.packs.length
                  ? esc(t('play.voicePacksInstalled', { packs: voiceStatus.packs.map(voicePackLabel).join(', ') }))
                  : esc(t('play.voicePacksNone'))
              }</div>
              ${voiceStatus && voiceStatus.dir ? `<div class="path">${esc(voiceStatus.dir)}</div>` : ''}
            </dd>
          </dl>
          <div class="row wrap" style="margin-top: var(--space-4)">
            <button class="btn" id="import-voice" ${ov.repo_ok ? '' : 'disabled'} title="${esc(t('play.importVoiceHint'))}">${icon('file-up')}${esc(t('play.importVoice'))}</button>
            <button class="btn" id="open-voice-dir" ${ov.repo_ok ? '' : 'disabled'}>${icon('folder-open')}${esc(t('play.openVoiceDir'))}</button>
          </div>
        </div>
      </div>`;
    refreshIcons();
    el.querySelector('#play')?.addEventListener('click', () => api('play').catch((e) => toast(errText(e), 'error')));
    el.querySelector('#finish-upgrade')?.addEventListener('click', () => ctx.go('setup'));
    el.querySelector('#record')?.addEventListener('click', async () => {
      try {
        await api('start_report');
        toast(t('play.recordStarted'));
      } catch (e) {
        toast(errText(e), 'error');
      }
    });
    el.querySelector('#to-setup')?.addEventListener('click', () => ctx.go('setup'));
    el.querySelector('#open-dir')?.addEventListener('click', () => api('open_game_dir').catch((e) => toast(errText(e), 'error')));
    el.querySelector('#manage-mods')?.addEventListener('click', () => ctx.go('mods'));
    el.querySelector('#open-mods')?.addEventListener('click', () => api('open_mods_dir').catch((e) => toast(errText(e), 'error')));
    el.querySelector('#import-saves')?.addEventListener('click', async () => {
      const folder = await api('pick_folder', { title: t('play.importSavesTitle'), start: null });
      if (!folder) return;
      try {
        toast(await api('import_saves', { path: folder }), 'ok');
      } catch (e) {
        toast(errText(e), 'error');
      }
    });
    el.querySelector('#import-voice')?.addEventListener('click', async () => {
      const file = await api('pick_voice_subtitle_file', { title: t('play.importVoiceTitle') });
      if (!file) return;
      try {
        const res = await api('import_voice_subtitles', { path: file });
        voiceStatus = res.status;
        toast(t('play.importVoiceDone', { packs: res.installed.map(voicePackLabel).join(', ') }), 'ok');
        draw();
      } catch (e) {
        toast(errText(e), 'error');
      }
    });
    el.querySelector('#open-voice-dir')?.addEventListener('click', () =>
      api('open_voice_subtitles_dir').catch((e) => toast(errText(e), 'error')),
    );
    el.querySelector('#add-steam')?.addEventListener('click', async () => {
      addingSteam = true;
      draw();
      try {
        toast(await api('add_to_steam'), 'ok');
      } catch (e) {
        toast(errText(e), 'error');
      }
      steam = await api('steam_status').catch(() => steam);
      addingSteam = false;
      draw();
    });
    el.querySelectorAll('[data-open]').forEach((b) =>
      b.addEventListener('click', () => api('open_game_file', { name: b.dataset.open }).catch((e) => toast(errText(e), 'error'))),
    );
  }

  ctx.onStatus((s) => {
    status = s;
    draw();
  });
  draw();
}
