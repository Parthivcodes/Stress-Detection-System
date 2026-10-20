import { X } from 'lucide-react';
import { motion } from 'framer-motion';
import { regions,metrics,displayValue,metricStatus } from '../data/wellness';
import type { Region,Period } from '../data/wellness';
import { StatusChip } from './StatusChip';
export function RegionDetailCard({region,period,index,onClose}:{region:Region;period:Period;index:number;onClose:()=>void}){
 const info=regions[region];
 return <motion.section initial={{opacity:0,y:12}} animate={{opacity:1,y:0}} exit={{opacity:0,y:8}} className="region-detail glass" aria-label={`${info.name} details`}><div className="flex items-center justify-between gap-2"><h3>{info.name}</h3><button className="icon-button" onClick={onClose} aria-label="Close region details"><X size={16}/></button></div><StatusChip status={period.regionStatus[region]}/><p>{info.subtitle}</p><dl>{info.metrics.map(id=>{const metric=metrics.find(m=>m.id===id)!;return <div key={id}><dt>{metric.name}</dt><dd>{displayValue(metric,index)} <small>{metric.unit}</small><span>{metricStatus(metric,index)}</span></dd></div>;})}</dl><small className="muted">Illustrative data · no diagnostic localization</small></motion.section>;
}
