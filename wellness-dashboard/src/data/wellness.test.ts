import { describe,it,expect } from 'vitest';
import { categories,periods,metrics,regions,filterMetrics,metricStatus,displayValue,diastolic } from './wellness';
describe('timeline data contract',()=>{
 it('has complete, finite readings and region states for every period',()=>{
  expect(new Set(metrics.map(m=>m.id)).size).toBe(metrics.length);
  metrics.forEach(metric=>{expect(metric.values).toHaveLength(periods.length);expect(metric.values.every(Number.isFinite)).toBe(true);});
  periods.forEach(period=>{expect(period.score).toBeGreaterThanOrEqual(0);expect(period.score).toBeLessThanOrEqual(100);expect(Object.keys(period.regionStatus).sort()).toEqual(Object.keys(regions).sort());});
  Object.values(regions).forEach(region=>region.metrics.forEach(id=>expect(metrics.some(m=>m.id===id)).toBe(true)));
 });
 it('covers recovery, elevated and critical mock states',()=>{
  expect(new Set(periods.map(p=>p.status))).toEqual(new Set(['Optimal','Elevated','Critical']));
  const cortisol=metrics.find(m=>m.id==='cortisol')!;
  expect(metricStatus(cortisol,0)).toBe('Optimal');expect(metricStatus(cortisol,2)).toBe('Elevated');expect(metricStatus(cortisol,3)).toBe('Critical');
  const hrv=metrics.find(m=>m.id==='hrv')!;expect(metricStatus(hrv,3)).toBe('Critical');expect(metricStatus(hrv,5)).toBe('Optimal');
 });
 it('keeps every category useful, while search spans categories',()=>{
  categories.forEach(c=>expect(filterMetrics(c,'',false).length).toBeGreaterThan(0));
  expect(filterMetrics('Hormone',' HRV ',false).map(m=>m.id)).toEqual(['hrv']);
  expect(filterMetrics('Hormone','',true)).toHaveLength(metrics.length);
  expect(filterMetrics('Sleep','no-such-value',false)).toEqual([]);
 });
 it('pairs blood-pressure readings with the selected period',()=>{
  const bp=metrics.find(m=>m.id==='bp')!;
  periods.forEach((_,i)=>expect(displayValue(bp,i)).toBe(`${bp.values[i]}/${diastolic[i]}`));
 });
});
