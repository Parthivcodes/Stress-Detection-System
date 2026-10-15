import { ArrowUpRight, ArrowDownRight, ArrowRight } from 'lucide-react';
import { GlassCard } from './GlassCard';
import { DotPlot } from './DotPlot';
import { StatusChip } from './StatusChip';
import { useAnimatedNumber } from '../hooks/useAnimatedNumber';
import { metricStatus,statusColors,diastolic } from '../data/wellness';
import type { Metric } from '../data/wellness';
export function MetricCard({metric,index}:{metric:Metric;index:number}){
 const value=useAnimatedNumber(metric.values[index]);const status=metricStatus(metric,index);const delta=metric.values[index]-metric.values[Math.max(0,index-1)];const Arrow=delta>0?ArrowUpRight:delta<0?ArrowDownRight:ArrowRight;
 return <GlassCard className={`metric-card ${metric.tall?'tall':''}`}><div className="metric-label"><i style={{background:statusColors[status]}}/>{metric.name}</div><div className="metric-value">{value.toFixed(metric.decimals)}{metric.id==='bp'&&<span className="bp-second">/{diastolic[index]}</span>}<small>{metric.unit}</small></div><div className="metric-delta" style={{color:statusColors[status]}}><span>{delta>0?'+':''}{delta.toFixed(metric.decimals)} {metric.unit}</span><Arrow size={19}/></div><DotPlot values={metric.values} index={index} color={statusColors[status]} label={metric.id} baseline={metric.baseline}/><StatusChip status={status} compact/><p>{status==='Optimal'?'Within optimal reference range':metric.note}</p></GlassCard>;
}
