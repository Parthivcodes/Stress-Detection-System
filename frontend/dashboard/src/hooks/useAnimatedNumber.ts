import { useEffect, useState } from 'react';
import { animate, useReducedMotion } from 'framer-motion';
export function useAnimatedNumber(value:number){
 const [display,setDisplay]=useState(value);const reduce=useReducedMotion();
 useEffect(()=>{if(reduce){setDisplay(value);return;}const controls=animate(display,value,{duration:.5,ease:'easeOut',onUpdate:setDisplay});return ()=>controls.stop();},[value,reduce]);
 return display;
}
