import { CheckCircle2, TriangleAlert, CircleAlert } from 'lucide-react';
import { statusColors } from '../data/wellness';
import type { Status } from '../data/wellness';
export function StatusChip({status,compact=false}:{status:Status;compact?:boolean}){const Icon=status==='Optimal'?CheckCircle2:status==='Elevated'?TriangleAlert:CircleAlert;return <span className={`status-chip ${compact?'compact':''}`} style={{color:statusColors[status]}}><Icon size={compact?11:13}/>{status}</span>;}
