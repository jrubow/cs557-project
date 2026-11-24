"""Reproducible descriptive behavior interpretation and reserved-test report."""
import argparse
from datetime import datetime,timezone
import html
import json
from pathlib import Path
import statistics
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from combat_rl.behavior import aggregate
from scripts.queue_optimization_windows import read_json,write_json,sha256
from scripts.run_test_evaluation_windows import validate_result


def fmt(value,digits=2):
    return 'not observed' if value is None else f'{value:.{digits}f}'


def table(headers,rows):
    return '<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+html.escape(str(h))+'</th>' for h in headers)+\
        '</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+\
        '</tbody></table></div>'


def derived(record):
    s=record['summary']; c=s['counts']; w=s['movement_windows']['retreat']; f=s['focus_fire']
    return dict(name=record['name'],steps=record['steps'],episodes=s['episodes'],kills=s['mean_kills'],
        seconds=s['mean_seconds'],aligned_percent=100*s['aligned_attack_fraction'] if s['aligned_attack_fraction'] is not None else None,
        yaw_correct_percent=100*s['yaw_correction_fraction'] if s['yaw_correction_fraction'] is not None else None,
        attack_requests=c['attack_pulses'],yaw_turns=c.get('yaw_turns',0),damage_per_attack=s['damage_per_attack_pulse'],
        crit_per100=s['critical_cues_per_100_attack_pulses'],crit_cues=c['critical_cues'],
        retreat_unanswered=w['unanswered_damage_windows'],retreat_windows=w['windows'],
        retreat_per_minute=1200*w['unanswered_damage_windows']/c['ticks'],
        near_wall_percent=100*s['near_wall_fraction'],kill_seconds=f['mean_kill_seconds'],
        kill_median=f['median_kill_seconds'],kill_completions=f['completed_zombies'],
        kill_incomplete=f['incomplete_damaged_zombies'],
        switch_percent=100*f['switch_before_kill_fraction'] if f['switch_before_kill_fraction'] is not None else None,
        focus_opportunities=f['counts'].get('focus_opportunities',0),
        multi_wounded_percent=100*f['multiple_wounded_fraction'] if f['multiple_wounded_fraction'] is not None else None)


def figures(history,batch_samples,test,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    blue,gray,orange='#226e89','#8a9299','#a76027'
    fig,axes=plt.subplots(2,3,figsize=(13,7.5),layout='constrained')
    panels=[('aligned_percent','Attacks inside geometric aiming band','Attack requests (%)'),
            ('retreat_per_minute','Unanswered retreat attacks','1-second windows / game minute'),
            ('crit_per100','Critical-attack sound cues','Cues / 100 attack requests'),
            ('kill_seconds','First damage to observed death','Mean game seconds; completed only'),
            ('switch_percent','Switching before the previous target dies','Eligible single-target transitions (%)'),
            ('near_wall_percent','Time near arena walls','Observed ticks within 2 blocks (%)')]
    for ax,(key,title,label) in zip(axes.flat,panels):
        values=[float('nan') if r[key] is None else r[key] for r in history]
        ax.axvspan(2,5.05,color=gray,alpha=.09)
        ax.plot([r['steps']/1e6 for r in history],values,color=blue,marker='o',linewidth=1.8)
        ax.set(title=title,xlabel='Total training decisions (millions)',ylabel=label,xlim=(0,5.1))
        ax.grid(alpha=.2)
    fig.suptitle('Observed behavior across C seed-0 checkpoints\n10 matched development scenarios per checkpoint; gray region: survival fine-tuning')
    fig.supxlabel('Descriptive proxies, not causal attribution. Kill-time means exclude incomplete timers; early samples include only 2–5 deaths.')
    fig.savefig(out/'behavior-interpretation.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    labels=[r['label'] for r in batch_samples]
    for ax,key,title in zip(axes,['mean_kills','mean_seconds'],['Mean zombies defeated','Mean game seconds survived']):
        x=list(range(len(labels)))
        ax.bar([i-.18 for i in x],[r['parent'][key] for r in batch_samples],.36,label='Original survival10/4M',color=gray)
        ax.bar([i+.18 for i in x],[r['selected'][key] for r in batch_samples],.36,label='Batch +1M',color=blue)
        ax.set(title=title,ylabel='Zombies / episode' if key=='mean_kills' else 'Game seconds / episode',
               xlabel='Separate matched scenario samples (20 episodes/model)',xticks=x,xticklabels=labels)
        ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Batch adaptation: evidence varies across samples\nFresh confirmation gain was below the predefined two-kill threshold')
    fig.savefig(out/'batch-evidence.png',dpi=180);plt.close(fig)
    if test is not None and test['complete']:
        fig,axes=plt.subplots(1,3,figsize=(13,4.8),layout='constrained')
        rows=test['runs']
        for ax,key,title,unit in zip(axes,['success','waves_completed','kills'],
                                    ['Five-wave completion','Completed waves','Zombies defeated'],
                                    ['Successful episodes (%)','Mean waves / episode','Mean kills / episode']):
            scale=100 if key=='success' else 1
            for group,offset,color in [('C 2M parent',-.18,gray),('survival10 4M',.18,blue)]:
                rr=sorted([r for r in rows if r['group']==group],key=lambda r:r['training_seed'])
                ax.bar([r['training_seed']+offset for r in rr],[scale*r[key] for r in rr],.36,label=group,color=color)
            extra=next(r for r in rows if r['group']=='batch +1M transfer')
            ax.bar([3.4],[scale*extra[key]],.36,label='Batch +1M (exploratory)',color=orange)
            ax.set(title=title,ylabel=unit,xlabel='Training seed / separate batch-transfer model',
                   xticks=[0,1,2,3.4],xticklabels=['Seed 0','Seed 1','Seed 2','Batch S0'])
            ax.grid(axis='y',alpha=.2)
        handles,labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='outside lower center',ncol=3,fontsize=9)
        fig.suptitle('Reserved five-wave test: 50 scenarios per model\nScenarios 200000–200049; policy choices locked before testing')
        fig.savefig(out/'reserved-test.png',dpi=180);plt.close(fig)


def build(args):
    outputs=args.outputs.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    behavior_path=outputs/'cs557-behavior-resumed-20261122/results.json'
    batch_path=outputs/'cs557-batch-finish-20261122b/results.json'
    test_root=outputs/'cs557-reserved-test-20261122'
    b,batch=read_json(behavior_path),read_json(batch_path)
    crit_root=outputs/'cs557-endless-calibration-recovery'
    controls=read_json(crit_root/'critical-controls.json')
    if not (b['critical_sensor_validated'] and controls['validated']):
        raise RuntimeError('Critical-cue interpretation requires validated controls')
    test=read_json(test_root/'results.json') if (test_root/'results.json').exists() else None
    complete=test is not None and test['complete'] and read_json(test_root/'status.json')['phase']=='complete'
    if not complete and not args.allow_pending:
        raise RuntimeError('Reserved test has not completed; use --allow-pending only for a labeled draft')
    history=[derived(r) for r in b['history']]
    by_name={r['name']:r for r in b['history']+b['references']+b['batch_and_transfer']}
    all_rows=[derived(r) for r in b['references']+b['history']+b['batch_and_transfer']]
    evidence=[];discrepancies=[];sensitivity=[];episode_rows=[]
    for record in by_name.values():
        directory=Path(record['output']);episodes=read_json(directory/'episodes.json')
        recomputed=aggregate(episodes)
        if any(record['summary'][key]!=value for key,value in recomputed.items()):
            raise RuntimeError('Behavior evidence does not reproduce summary')
        meta=read_json(directory/'metadata.json')
        if sha256(directory/'ticks.csv.gz')!=meta['trace_sha256']:
            raise RuntimeError('Behavior trace identity changed')
        evidence.append(dict(name=record['name'],source_revision=meta['runtime']['git_revision'],
            model_sha256=record['model_sha256'],episodes_sha256=sha256(directory/'episodes.json'),
            metadata_sha256=sha256(directory/'metadata.json'),trace_sha256=meta['trace_sha256']))
        kept=[]
        for ep in episodes:
            focus=ep['focus_fire']; deaths=[z for z in focus['zombies'] if z['death_tick'] is not None]
            mismatch=len(deaths)!=ep['metrics']['kills']
            if mismatch:
                discrepancies.append(dict(evaluation=record['name'],scenario=ep['scenario_seed'],
                    health_zero_deaths=len(deaths),player_kills=ep['metrics']['kills'],
                    deaths_on_terminal_tick=sum(z['death_tick']==ep['counts']['ticks'] for z in deaths),
                    episode_terminal_reason=ep['metrics']['terminal_reason']))
            for z in deaths:
                if z['kill_seconds'] is not None and not (mismatch and z['death_tick']==ep['counts']['ticks']):
                    kept.append(z['kill_seconds'])
            kiting=[w for w in ep['windows'] if not w['partial'] and w['group']=='retreat']
            episode_rows.append(dict(evaluation=record['name'],scenario=ep['scenario_seed'],metrics=ep['metrics'],
                retreat_windows=len(kiting),unanswered_retreat_windows=sum(w['damage_dealt']>0 and w['damage_taken']==0 for w in kiting),
                focus_fire=aggregate([ep])['focus_fire']))
        sensitivity.append(dict(evaluation=record['name'],original_mean=record['summary']['focus_fire']['mean_kill_seconds'],
            excluding_terminal_mismatch_mean=statistics.mean(kept) if kept else None,retained_deaths=len(kept)))
    parent=by_name['batch-parent']['summary'];selected=by_name['batch-selected']['summary']
    batch_samples=[dict(label='Development',parent=batch['baseline'],selected=batch['selected_development']),
        dict(label='Confirmation',**batch['confirmation']),
        dict(label='Behavior sample',parent=dict(mean_kills=parent['mean_kills'],mean_seconds=parent['mean_seconds']),
             selected=dict(mean_kills=selected['mean_kills'],mean_seconds=selected['mean_seconds']))]
    figures(history,batch_samples,test if complete else None,out)
    write_json(out/'behavior-summary.json',dict(records=all_rows,definitions=b['definitions'],evidence=evidence,
        critical_sensor_validated=b['critical_sensor_validated'],
        health_vs_player_kill_discrepancies=discrepancies,kill_time_sensitivity=sensitivity))
    write_json(out/'batch-evidence.json',dict(samples=batch_samples,task=batch['task'],
        improvement_confirmed=batch['improvement_confirmed'],selected_model=batch['selected'],
        selected_sha256=batch['selected_sha256'],five_wave_transfer=batch['five_wave_transfer']))
    for name in ('critical-controls.json','critical-history.json'):
        (out/name).write_bytes((crit_root/name).read_bytes())
    write_json(out/'episode-summary.json',episode_rows)
    manifest=read_json(test_root/'manifest.json')
    write_json(out/'test-manifest.json',manifest)
    write_json(out/'final-policies.json',dict(selection_locked_before_test=True,
        selection_frozen_utc=manifest['frozen_utc'],
        primary_recipe=[r for r in manifest['jobs'] if r['group']=='survival10 4M'],
        demonstration_model=manifest['demonstration_model'],
        exploratory_batch=[r for r in manifest['jobs'] if r['group']=='batch +1M transfer'],
        batch_upgrade_confirmed=manifest['batch_upgrade_confirmed']))
    if complete:
        status=read_json(test_root/'status.json')
        if (len(test['runs'])!=7 or test['episodes']!=350 or test['selection_changed'] or test['policy_updated']
                or not status['models_unchanged'] or not status['selection_unchanged']):
            raise RuntimeError('Reserved-test completion or policy lock mismatch')
        jobs={job['name']:job for job in manifest['jobs']}
        if set(jobs)!={run['name'] for run in test['runs']}:
            raise RuntimeError('Reserved-test model set mismatch')
        for path,digest in manifest['development_evidence_sha256'].items():
            if sha256(path)!=digest:
                raise RuntimeError('Pre-test development evidence changed')
        source=Path(manifest['source'])
        if (subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()!=manifest['source_revision']
                or subprocess.check_output(['git','status','--porcelain'],cwd=source,text=True).strip()):
            raise RuntimeError('Frozen evaluation source changed')
        write_json(out/'reserved-test-results.json',test)
        test_evidence=[]
        for run in test['runs']:
            job=jobs[run['name']];model=Path(job['model'])
            if sha256(model)!=job['model_sha256'] or sha256(model.with_suffix('.json'))!=job['metadata_sha256']:
                raise RuntimeError('Locked model or metadata changed')
            src=Path(run['test_csv'])
            recomputed=validate_result(src,job,read_json(model.with_suffix('.json'))['config'],manifest['first_seed'])
            if any(run[key]!=value for key,value in recomputed.items()):
                raise RuntimeError('Reserved-test summary does not reproduce episode records')
            receipt=read_json(test_root/'logs'/(run['name']+'.status.json'))
            if (receipt['exit_code']!=0 or receipt['child_exit_code']!=0
                    or receipt['remaining_descendant_pids'] or receipt.get('supervisor_errors')):
                raise RuntimeError('Unclean evaluation supervisor')
            if sha256(src)!=run['csv_sha256'] or sha256(src.with_suffix('.json'))!=run['metadata_sha256']:
                raise RuntimeError('Reserved test CSV or metadata changed')
            (out/(run['name']+'.csv')).write_bytes(src.read_bytes())
            (out/(run['name']+'.json')).write_bytes(src.with_suffix('.json').read_bytes())
            test_evidence.append(dict(name=run['name'],csv_sha256=sha256(src),
                                      metadata_sha256=sha256(src.with_suffix('.json')),supervisor=receipt))
    else:
        test_evidence=[]
    write_json(out/'provenance.json',dict(generated_utc=datetime.now(timezone.utc).isoformat(),
        report_script_sha256=sha256(__file__),
        behavior_results_sha256=sha256(behavior_path),batch_results_sha256=sha256(batch_path),
        critical_evidence_sha256={name:sha256(crit_root/name) for name in ('critical-controls.json','critical-history.json')},
        test_manifest_sha256=sha256(test_root/'manifest.json'),test_evidence=test_evidence,
        test_results_sha256=sha256(test_root/'results.json') if complete else None,
        test_complete=complete,source_files=[str(behavior_path),str(batch_path),str(test_root)]))
    h={r['name']:r for r in history};early=h['history-c100000'];mid=h['history-c2000000'];final=h['history-survival10_4000000']
    fs=by_name['history-survival10_4000000']['summary']
    findings=[
      ('Aiming: clear improvement in the geometric proxy',
       f"The share of attack requests in the aiming band rises from {early['aligned_percent']:.1f}% at 102k decisions to {final['aligned_percent']:.1f}% at 4M. "
       f"At 4M, {fs['counts']['yaw_improving_turns']:,}/{fs['counts']['yaw_turns']:,} measured turns reduce yaw error by more than one degree ({final['yaw_correct_percent']:.1f}%). "
       f"Damage per attack request rises from {early['damage_per_attack']:.2f} to {final['damage_per_attack']:.2f} HP. "
       "This supports better aiming/attack coordination. The band targets the nearest visible torso and is not an exact entity raycast; initial action bias already favored steady aim and attacking."),
      ('Hit-and-retreat: strong descriptive evidence of kiting-like behavior',
       f"No qualifying unanswered retreat windows occur through the 297k checkpoint. At 502k, seven appear across six of ten episodes. At 4M, {final['retreat_unanswered']} of {final['retreat_windows']} retreat windows deal damage with none received, across all ten episodes. "
       f"Their rate increases from {mid['retreat_per_minute']:.2f}/game minute at 2M to {final['retreat_per_minute']:.2f} at 4M. "
       "These windows require backward commands, measured player motion away from a nearby visible zombie, and an attack. Enemy knockback alone does not qualify. They support hit-and-retreat behavior, not proof of deliberate reach management or invulnerability."),
      ('Critical hits: behavior emerges and becomes common',
       f"The scripted sensor controls passed. Critical sound cues are absent at the sampled 10k/51k checkpoints and appear at 102,400 decisions in both the earlier crit study and this separate behavior pass. "
       f"At 4M there are {final['crit_cues']} cues per {final['attack_requests']:,} attack requests ({final['crit_per100']:.1f}/100), versus {early['crit_per100']:.2f}/100 at 102k. "
       "This is the earliest sampled evidence, not the exact learning step. Cues are deduplicated nearby sounds rather than server-side hit events, and the policy receives no explicit crit observation or bonus."),
      ('Focus fire: faster observed kills, but no consistent single-target discipline',
       f"Mean first-damage-to-observed-death time decreases from {early['kill_seconds']:.2f}s at 102k ({early['kill_completions']} completed, {early['kill_incomplete']} incomplete) to {final['kill_seconds']:.2f}s at 4M ({final['kill_completions']} completed, {final['kill_incomplete']} incomplete). "
       f"However, the 4M policy switches before the previous target dies in {fs['focus_fire']['counts']['switches_before_kill']}/{final['focus_opportunities']} eligible transitions ({final['switch_percent']:.1f}%). "
       f"Multiple zombies are wounded during {final['multi_wounded_percent']:.1f}% of multi-enemy ticks, versus {early['multi_wounded_percent']:.1f}% at 102k. "
       "Faster kills therefore do not establish improved focus fire. Later models face more enemies/waves; sweeps and environmental damage complicate attribution. Multi-target damage ticks are excluded from ordered target-switch transitions."),
      ('Walls and attack timing: progress with remaining weaknesses',
       f"The 4M policy spends {final['near_wall_percent']:.1f}% of ticks near walls, down from {early['near_wall_percent']:.1f}% at 102k, but still most of the episode. "
       f"It escapes {fs['wall_escapes']} observed wall intervals, while {fs['wall_censored']} end unfinished; only {fs['wall_escapes_within_5_seconds']} escape occurs within five seconds. "
       f"Incoming damage is {fs['near_wall_damage_per_second']:.3f} HP/s near walls versus {fs['away_wall_damage_per_second']:.3f} away, a descriptive association with different combat exposure. "
       "Median attack-request spacing is 12 game ticks (0.6s) at both 102k and 4M, so later efficiency is not simply a faster attack-request cadence. Enemy reacquisition does not show a consistent monotonic improvement.")]
    batch_text=(f"The selected batch model gains {batch['selected_development']['mean_kills']-batch['baseline']['mean_kills']:.2f} kills on the development sample, "
       f"but only {batch['confirmation']['selected']['mean_kills']-batch['confirmation']['parent']['mean_kills']:.2f} on fresh confirmation (19.1 versus 17.7), below the predefined two-kill threshold. "
       "The later behavior sample is favorable again (21.55 versus 18.85 kills), but cannot retroactively replace the confirmation rule. "
       f"Batch kill time changes little ({parent['focus_fire']['mean_kill_seconds']:.2f}s to {selected['focus_fire']['mean_kill_seconds']:.2f}s), while switching increases "
       f"({100*parent['focus_fire']['switch_before_kill_fraction']:.1f}% to {100*selected['focus_fire']['switch_before_kill_fraction']:.1f}%). "
       "The fixed recordings show 12, 24 and 42 kills; 42 is the strongest demonstration, not an average. Keep this model as an exploratory batch demo.")
    limitations=("These are descriptive observations from one training-seed lineage across checkpoints, with ten matched scenarios each; batch comparisons use twenty. "
       "Minecraft AI randomness is not bit-exact. Checkpoints visit different health, wave and enemy-count contexts. Survival fine-tuning changes several settings together at 2M. "
       "No ablation establishes causal acquisition, intention or optimality. Means pool eligible opportunities; per-episode evidence and denominators accompany them. "
       "Three episodes record one more health-zero death than the player's kill counter, always on the player-death tick. "
       "This is consistent with update timing or attribution differences, not proof of a player kill. Those primary clocks are labeled observed deaths; excluding the three ambiguous terminal deaths changes each affected mean by less than 0.005 seconds. "
       "There are no lost-entity or initially-injured exclusions in these particular traces; unfinished damaged zombies are still excluded from completed-only means and counted separately. "
       "Startup recovery reused validated completed outputs across two operational source revisions. The Windows cleanup adjustment changes startup handling, not combat inputs or rewards.")
    test_text='Reserved test is still running; no final test conclusion is available in this draft.'
    test_rows=[]
    if complete:
        groups={g['group']:g for g in test['groups']};old=groups['C 2M parent'];new=groups['survival10 4M']
        extra=next(r for r in test['runs'] if r['group']=='batch +1M transfer')
        demo=next(r for r in test['runs'] if r['group']=='survival10 4M' and r['training_seed']==0)
        paired=', '.join(f"seed {r['seed']}: {r['waves_completed']:+.2f}" for r in test['paired_seed_changes'])
        test_text=(f"All 350 reserved episodes completed. The three survival10/4M models won {new['successes']}/150 episodes "
           f"({100*new['success_mean']:.1f}% mean success), versus {old['successes']}/150 for their C/2M parents ({100*old['success_mean']:.1f}%). "
           f"Mean completed waves were {new['waves_completed_mean']:.2f} ± {new['waves_completed_sd']:.2f} versus {old['waves_completed_mean']:.2f} ± {old['waves_completed_sd']:.2f}; "
           f"mean kills {new['kills_mean']:.2f} ± {new['kills_sd']:.2f} versus {old['kills_mean']:.2f} ± {old['kills_sd']:.2f} (mean ± sample SD across the three trained models). "
           f"Paired changes in waves were {paired}. "
           f"The preselected seed-0 demonstration model completed all five waves in {demo['successes']}/50 tests ({100*demo['success']:.0f}%). "
           f"The separate batch +1M model achieved {extra['successes']}/50 ({100*extra['success']:.0f}%) on this five-wave transfer test. "
           "A single 50-episode comparison is insufficient to establish consistent superiority of that batch model; it does not change the pre-test selection. "
           "These are combined extra-training/settings effects, not a controlled reward-only ablation. Three training-seed lineages provide limited evidence about run-to-run variability. "
           "All policies used the same 50 reserved spawn/yaw scenarios in the original arena; this does not test new worlds, equipment or enemy types. "
           "The pre-test policy choices remain unchanged, and the reserved scenarios are now consumed: further optimization needs a new untouched evaluation set.")
        for r in test['runs']:
            test_rows.append([r['name'],f"{r['successes']}/50",f"{100*r['success']:.1f}%",fmt(r['waves_completed']),fmt(r['kills']),fmt(r['elapsed_seconds'])])
    behavior_rows=[[r['name'],r['episodes'],fmt(r['aligned_percent'],1),fmt(r['retreat_per_minute']),fmt(r['crit_per100']),
                    fmt(r['kill_seconds']),f"{r['kill_completions']} / {r['kill_incomplete']}",fmt(r['switch_percent'],1)] for r in all_rows]
    markdown=['# Final policy and behavior study',
      '**Status: '+('reserved test complete.' if complete else 'development interpretation; reserved test still running.')+'**',
      'The final five-wave recipe is **survival10/4M across seeds 0–2**, with seed 0 selected for demonstration before testing. '
      'The **batch +1M seed-0 model** remains a separate exploratory demonstration policy. See [the pre-test selection](SELECTION.md).',
      '## Reserved five-wave test',test_text]
    if complete:
        test_table=[
          '| Model | Wins | Success | Mean waves | Mean kills | Mean game seconds |',
          '| --- | ---: | ---: | ---: | ---: | ---: |']
        test_table+=['| '+' | '.join(map(str,row))+' |' for row in test_rows]
        markdown+=['![Reserved test results](reserved-test.png)','\n'.join(test_table)]
    markdown+=['## Behavior interpretation','![Behavior across checkpoints](behavior-interpretation.png)']
    for title,body in findings:markdown+=['### '+title,body]
    markdown+=['## Batch adaptation',batch_text,'![Batch evidence](batch-evidence.png)',
               '## Limits and measurement audit',limitations,
               '## Reproducible evidence',
               'The [compact behavior data](behavior-summary.json), [episode summaries](episode-summary.json), '
               '[critical sensor controls](critical-controls.json), [earlier crit history](critical-history.json), [batch evidence](batch-evidence.json), '
               '[frozen test manifest](test-manifest.json), [final policy paths and hashes](final-policies.json), and [hash provenance](provenance.json) accompany this report. '
               'Per-model reserved-test CSVs, evaluation metadata and `reserved-test-results.json` are exported when complete. '
               'Full checkpoints, raw traces and videos remain outside Git in the project output directories.',
               'Rebuild with `python scripts/report_final_study.py --outputs <project-outputs> --output reports/final-study-20261122`.']
    (out/'README.md').write_text('\n\n'.join(markdown)+'\n',encoding='utf-8')
    sections='<section><h2>Reserved five-wave test</h2><p>'+html.escape(test_text)+'</p>'
    if complete:sections+='<img src="reserved-test.png" alt="Reserved test results by training seed">'+table(['Model','Wins','Success','Mean waves','Mean kills','Game seconds'],test_rows)
    sections+='</section><section><h2>What the behavior evidence supports</h2><img src="behavior-interpretation.png" alt="Behavior metrics over training">'
    for title,body in findings:sections+='<h3>'+html.escape(title)+'</h3><p>'+html.escape(body)+'</p>'
    sections+='<details><summary>Inspect all checkpoint and reference measurements</summary>'+table(
        ['Evaluation','Episodes','Aligned attacks %','Retreat windows/min','Crit cues/100','Mean kill seconds','Completed / incomplete','Switch %'],behavior_rows)+'</details></section>'
    sections+='<section><h2>Batch adaptation</h2><p>'+html.escape(batch_text)+'</p><img src="batch-evidence.png" alt="Separate batch samples"></section>'
    sections+='<section><h2>Limits and measurement audit</h2><p>'+html.escape(limitations)+'</p></section>'
    sections+='<section><h2>Evidence files</h2><p><a href="SELECTION.md">Pre-test policy lock</a> · <a href="final-policies.json">Final policy paths and hashes</a> · <a href="README.md">Markdown report</a> · <a href="behavior-summary.json">Behavior data</a> · <a href="episode-summary.json">Episode evidence</a> · <a href="test-manifest.json">Test manifest</a> · <a href="provenance.json">Provenance</a></p></section>'
    page='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CS557 — Final policies and behavior evidence</title><style>
:root{color-scheme:light}*{box-sizing:border-box}body{margin:0;background:#f6f7f8;color:#1c2932;font:16px/1.65 system-ui,Arial,sans-serif}
main{max-width:1120px;margin:auto;padding:36px 28px}header{border-bottom:2px solid #226e89;padding-bottom:24px}h1{font-size:30px;line-height:1.2;margin:8px 0 18px}h2{font-size:24px;margin-top:0}h3{font-size:18px;color:#226e89;margin-bottom:8px}p{margin:8px 0 22px}section{padding:28px 0;border-bottom:1px solid #d5dce0}.eyebrow{font-size:13px;letter-spacing:.1em;text-transform:uppercase;color:#536570}.decision{font-size:19px;max-width:950px}img{display:block;width:100%;height:auto;background:white;margin:20px 0}a{color:#175d77}details{margin-top:24px}summary{cursor:pointer;font-weight:650}.table-wrap{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px;margin-top:16px}th,td{padding:9px 12px;border-bottom:1px solid #d5dce0;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}th{background:#e9eff2}footer{font-size:13px;color:#536570;margin-top:22px}@media(max-width:650px){main{padding:24px 16px}h1{font-size:26px}.decision{font-size:17px}}
</style></head><body><main><header><div class="eyebrow">CS557 · Minecraft reinforcement learning · November 22, 2026</div>
<h1>Final policies and learned-behavior evidence</h1><p class="decision"><strong>Five-wave policy:</strong> survival10 at 4M decisions, retaining all three training seeds. <strong>Demonstration:</strong> seed 0. <strong>Batch demo:</strong> +1M adaptation, kept exploratory.</p></header>'''+sections+\
        '<footer>Local, self-contained report. All figures use project measurements; no external scripts or network dependencies.</footer></main></body></html>'
    (out/'report.html').write_text(page,encoding='utf-8')
    print(json.dumps(dict(output=str(out),test_complete=complete,behavior_evaluations=len(all_rows),episodes=len(episode_rows),measurement_discrepancies=len(discrepancies))))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outputs',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--allow-pending',action='store_true')
    build(parser.parse_args())


if __name__=='__main__':
    main()
