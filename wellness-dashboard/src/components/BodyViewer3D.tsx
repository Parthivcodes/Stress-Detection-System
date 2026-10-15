import { Component, useEffect, useMemo, useRef } from 'react';
import type { ReactNode } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';
import { statusColors } from '../data/wellness';
import type { Region,Period } from '../data/wellness';
type Props={period:Period;selected:Region|null;onSelect:(region:Region)=>void;zoom:number;rotate:boolean;reset:number;flat:boolean};
class RenderBoundary extends Component<{children:ReactNode},{failed:boolean}>{state={failed:false};static getDerivedStateFromError(){return {failed:true};}render(){return this.state.failed?<div className="body-fallback"><span>◯</span><span>╱│╲</span><span>│</span><p>3D preview unavailable.<br/>Use the region buttons to explore the same data.</p></div>:this.props.children;}}
function Shape({position,scale,color='#dfe4df',opacity=.14,detail=2,onClick}:{position:[number,number,number];scale:[number,number,number];color?:string;opacity?:number;detail?:number;onClick?:()=>void}){
 return <mesh position={position} scale={scale} onClick={onClick?e=>{e.stopPropagation();onClick();}:undefined}><icosahedronGeometry args={[1,detail]}/><meshBasicMaterial color={color} wireframe transparent opacity={opacity} depthWrite={false}/></mesh>;
}
function Rib({points}:{points:[number,number,number][]}){
 const geometry=useMemo(()=>new THREE.TubeGeometry(new THREE.CatmullRomCurve3(points.map(p=>new THREE.Vector3(...p))),36,.008,3,false),[points]);
 useEffect(()=>()=>geometry.dispose(),[geometry]);
 return <mesh geometry={geometry}><meshBasicMaterial color="#e3e8e1" transparent opacity={.44}/></mesh>;
}
function Figure({period,onSelect,selected,rotate}:Props){
 const group=useRef<THREE.Group>(null);
 useFrame(({clock})=>{if(group.current)group.current.rotation.y=rotate?Math.sin(clock.elapsedTime*.16)*.10:0;});
 const color=(r:Region)=>statusColors[period.regionStatus[r]];
 const opacity=(r:Region)=>selected===r ? .7 : .32;
 const ribs=useMemo(()=>Array.from({length:10},(_,i)=>{const y=1.02-i*.12;const width=.52+Math.sin((i+1)/11*Math.PI)*.28;return [-1,1].map(side=>[[0,y,-.21],[side*width*.8,y+.04,-.25],[side*width,y-.02,.05],[side*width*.7,y-.1,.35],[0,y-.14,.37]] as [number,number,number][]);}).flat(),[]);
 return <group ref={group} position={[0,-.03,0]}>
 <Shape position={[0,2.04,0]} scale={[.48,.61,.43]} opacity={.26}/><Shape position={[0,1.66,.10]} scale={[.32,.27,.31]} opacity={.3} detail={1}/>
 <Shape position={[0,2.15,.02]} scale={[.37,.36,.32]} color={color('head')} opacity={opacity('head')} onClick={()=>onSelect('head')}/>
 <Shape position={[0,1.35,0]} scale={[.19,.25,.2]} opacity={.3} detail={1}/>
 <Shape position={[0,.25,-.03]} scale={[.86,.99,.40]} opacity={.095}/><Shape position={[0,-.83,0]} scale={[.60,.40,.34]} opacity={.13}/>
 {ribs.map((points,i)=><Rib key={i} points={points}/>)}
 {Array.from({length:22},(_,i)=><Shape key={i} position={[0,1.42-i*.11,-.19]} scale={[.11,.05,.10]} opacity={.55} detail={0}/>)}
 <Rib points={[[0,1.11,.35],[0,.4,.4],[0,-.13,.28]]}/>
 {[-1,1].map(side=><group key={side}>
 <Rib points={[[0,1.1,.2],[side*.48,1.2,.08],[side*.91,1.06,0]]}/>
 <Shape position={[side*.91,1.02,0]} scale={[.28,.27,.27]} color={color('shoulders')} opacity={opacity('shoulders')} onClick={()=>onSelect('shoulders')}/>
 <group position={[side*1.06,.45,0]} rotation={[0,0,side*.2]}><Shape position={[0,0,0]} scale={[.19,.55,.18]}/><mesh><cylinderGeometry args={[.045,.05,.94,7]}/><meshBasicMaterial color="#e0e4dc" wireframe transparent opacity={.4}/></mesh></group>
 <Shape position={[side*1.18,-.1,.04]} scale={[.16,.17,.17]} opacity={.3} detail={1}/>
 <group position={[side*1.22,-.64,.08]} rotation={[0,0,side*.06]}><Shape position={[0,0,0]} scale={[.14,.49,.12]}/><mesh><cylinderGeometry args={[.035,.03,.9,7]}/><meshBasicMaterial color="#dce3d8" wireframe transparent opacity={.4}/></mesh></group>
 <Shape position={[side*1.27,-1.29,.1]} scale={[.13,.22,.08]} opacity={.25} detail={1}/>
 <Shape position={[side*.33,.51,.06]} scale={[.26,.45,.23]} opacity={.10}/>
 </group>)}
 <Shape position={[.18,.43,.35]} scale={[.20,.28,.18]} color={color('heart')} opacity={opacity('heart')+.1} onClick={()=>onSelect('heart')}/>
 <mesh position={[.18,.43,.35]} onClick={e=>{e.stopPropagation();onSelect('heart');}}><sphereGeometry args={[.15,16,16]}/><meshBasicMaterial color={color('heart')} transparent opacity={.12}/></mesh>
 <Shape position={[0,-.44,.19]} scale={[.43,.29,.25]} color={color('gut')} opacity={opacity('gut')} onClick={()=>onSelect('gut')}/>
 <mesh position={[0,2.1,0]} onClick={e=>{e.stopPropagation();onSelect('head');}}><sphereGeometry args={[.5,8,8]}/><meshBasicMaterial transparent opacity={0} depthWrite={false}/></mesh>
 </group>;
}
function CameraSettings({zoom,reset,flat}:{zoom:number;reset:number;flat:boolean}){
 const {camera}=useThree();
 useEffect(()=>{camera.position.set(flat?0:.25,.5,zoom);camera.lookAt(0,.5,0);camera.updateProjectionMatrix();},[camera,zoom,reset,flat]);return null;
}
export function BodyViewer3D(props:Props){
 return <RenderBoundary><Canvas camera={{position:[.25,.5,8.6],fov:36}} dpr={[1,1.5]} gl={{alpha:true,antialias:true,powerPreference:'low-power'}} aria-label="Interactive conceptual wireframe upper body" fallback={<div className="body-fallback">3D unavailable. Explore regions using the labeled buttons.</div>}><CameraSettings zoom={props.zoom} reset={props.reset} flat={props.flat}/><Figure {...props}/><OrbitControls key={`${props.reset}-${props.flat}`} target={[0,.5,0]} enablePan={false} enableZoom={false} enableRotate={!props.flat} minPolarAngle={Math.PI*.35} maxPolarAngle={Math.PI*.65} minAzimuthAngle={-1.3} maxAzimuthAngle={1.3}/></Canvas></RenderBoundary>;
}
