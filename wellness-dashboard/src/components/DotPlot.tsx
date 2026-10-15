import { useState } from 'react';
import { periods } from '../data/wellness';
export function DotPlot({values,index,color,label,baseline}:{values:number[];index:number;color:string;label:string;baseline:number}){
 const [hover,setHover]=useState<number|null>(null);
 const low=Math.min(...values,baseline)*.8,high=Math.max(...values,baseline)*1.15;
 const y=(v:number)=>65-(v-low)/(high-low)*48;
 const point=hover??index;
 return <div className="dotplot"><svg viewBox="0 0 200 88" role="group" aria-label={`${label} history, May to October 2026`}>
 <defs><linearGradient id={`fade-${label.replace(/\W/g,'')}`} x1="0" y1="0" x2="0" y2="1"><stop stopColor={color} stopOpacity=".13"/><stop offset="1" stopColor={color} stopOpacity=".01"/></linearGradient></defs>
 <path d={`M8 ${y(baseline)}H192V80H8Z`} fill={`url(#fade-${label.replace(/\W/g,'')})`}/><path d={`M8 ${y(baseline)}H192`} stroke="#a4ac9b" strokeWidth=".7"/>
 {values.map((value,i)=><g key={i} tabIndex={0} role="img" aria-label={`${periods[i].label}: ${value}${i===index?', selected period':''}`} onFocus={()=>setHover(i)} onBlur={()=>setHover(null)} onMouseEnter={()=>setHover(i)} onMouseLeave={()=>setHover(null)}><circle cx={16+i*33} cy={y(value)} r="11" fill="transparent"/><circle cx={16+i*33} cy={y(value)} r={i===index?4:3} fill={i===index?color:'#dee2dd'} opacity={i>index?.35:1}/>{i===index&&<circle cx={16+i*33} cy={y(value)} r="9" stroke={color} fill="none" strokeWidth="1"/>}</g>)}
 </svg><span className={`plot-tooltip ${hover===null?'quiet':''}`}>{periods[point].month} · {values[point]} {hover===null?'· selected':''}</span></div>;
}
