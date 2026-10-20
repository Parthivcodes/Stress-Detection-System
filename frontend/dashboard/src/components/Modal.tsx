import { useEffect, useRef } from 'react';
import type { PropsWithChildren } from 'react';
import { X } from 'lucide-react';
export function Modal({title,onClose,children}:PropsWithChildren<{title:string;onClose:()=>void}>){
 const ref=useRef<HTMLDialogElement>(null);
 useEffect(()=>{const before=document.activeElement as HTMLElement|null;const dialog=ref.current;dialog?.showModal();const original=document.body.style.overflow;document.body.style.overflow='hidden';return()=>{dialog?.close();document.body.style.overflow=original;before?.focus();};},[]);
 return <dialog ref={ref} className="modal glass" aria-labelledby="modal-title" onCancel={e=>{e.preventDefault();onClose();}} onClick={e=>{if(e.target===e.currentTarget)onClose();}}><div className="modal-inner"><div className="flex items-center justify-between gap-4"><h2 id="modal-title">{title}</h2><button className="icon-button" onClick={onClose} aria-label="Close dialog"><X size={18}/></button></div>{children}</div></dialog>;
}
