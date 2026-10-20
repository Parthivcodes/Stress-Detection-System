import type { PropsWithChildren } from 'react';
import { motion, useReducedMotion } from 'framer-motion';
export function GlassCard({children,className=''}:PropsWithChildren<{className?:string}>){const reduce=useReducedMotion();return <motion.div whileHover={reduce?undefined:{y:-3}} className={`glass ${className}`}>{children}</motion.div>;}
