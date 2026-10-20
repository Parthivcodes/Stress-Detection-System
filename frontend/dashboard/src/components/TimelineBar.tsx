import { CalendarDays } from 'lucide-react';
import { periods } from '../data/wellness';
import { StatusChip } from './StatusChip';
export function TimelineBar({index,onChange}:{index:number;onChange:(value:number)=>void}){
 return <section className="timeline glass" aria-label="Wellness history timeline"><div className="timeline-title"><CalendarDays size={17}/><span>Your timeline<small>2026 · mock history</small></span></div><div className="timeline-main"><div className="timeline-markers">{periods.map((period,i)=><button key={period.label} onClick={()=>onChange(i)} className={index===i?'selected':''} aria-pressed={index===i} aria-label={`${period.label}, ${period.status}`}><span className="period-label">{period.month}<small>{period.year}</small></span><span className="period-dots" aria-hidden="true">•••<i/></span><StatusChip status={period.status} compact/></button>)}</div><input aria-label="Scrub wellness timeline" aria-valuetext={`${periods[index].label}, ${periods[index].status}`} type="range" min="0" max={periods.length-1} value={index} onChange={e=>onChange(Number(e.target.value))}/></div></section>;
}
