export type Status = 'Optimal' | 'Elevated' | 'Critical';
export type Category = 'Hormone' | 'Cardiovascular' | 'Nervous System' | 'Sleep' | 'Mood' | 'Digestive';
export type Region = 'head' | 'shoulders' | 'heart' | 'gut';
export type Metric = {id:string; name:string; unit:string; category:Category[]; values:number[]; decimals:number; baseline:number; direction:'up'|'down'; note:string; tall?:boolean};
export type Period = {label:string;month:string;year:string;score:number;status:Status;weight:number;regionStatus:Record<Region,Status>};
export const user = {name:'Alex Morgan',id:'NV-7294',dob:'18 Jul 1992',age:34,gender:'Female',ethnicity:'Other',counselors:['Julian Hayes','Avery Brooks','Sam Rivera']};
export const categories:Category[]=['Hormone','Cardiovascular','Nervous System','Sleep','Mood','Digestive'];
export const periods:Period[]=[
 {label:'May 2026',month:'May',year:'2026',score:24,status:'Optimal',weight:63.2,regionStatus:{head:'Optimal',shoulders:'Optimal',heart:'Optimal',gut:'Optimal'}},
 {label:'Jun 2026',month:'Jun',year:'2026',score:38,status:'Optimal',weight:63.4,regionStatus:{head:'Optimal',shoulders:'Elevated',heart:'Optimal',gut:'Optimal'}},
 {label:'Jul 2026',month:'Jul',year:'2026',score:61,status:'Elevated',weight:63.6,regionStatus:{head:'Elevated',shoulders:'Elevated',heart:'Elevated',gut:'Optimal'}},
 {label:'Aug 2026',month:'Aug',year:'2026',score:84,status:'Critical',weight:63.5,regionStatus:{head:'Critical',shoulders:'Elevated',heart:'Critical',gut:'Elevated'}},
 {label:'Sep 2026',month:'Sep',year:'2026',score:49,status:'Elevated',weight:63.3,regionStatus:{head:'Elevated',shoulders:'Elevated',heart:'Optimal',gut:'Optimal'}},
 {label:'Oct 2026',month:'Oct',year:'2026',score:29,status:'Optimal',weight:63.4,regionStatus:{head:'Optimal',shoulders:'Optimal',heart:'Optimal',gut:'Optimal'}}
];
// All values, classifications and notes are fictional UI fixtures, not reference ranges.
export const metrics:Metric[]=[
 {id:'cortisol',name:'Cortisol (AM)',unit:'µg/dL',category:['Hormone'],values:[12.1,15.2,20.8,26.4,17.6,13.2],decimals:1,baseline:14,direction:'up',note:'Approaching upper reference limit'},
 {id:'dhea',name:'DHEA-S',unit:'µg/dL',category:['Hormone'],values:[190,181,142,105,160,185],decimals:0,baseline:185,direction:'down',note:'Lower than previous baseline',tall:true},
 {id:'glucose',name:'Glucose',unit:'mg/dL',category:['Hormone','Digestive'],values:[88,91,103,116,96,90],decimals:0,baseline:90,direction:'up',note:'Higher than previous baseline'},
 {id:'insulin',name:'Insulin',unit:'µIU/mL',category:['Hormone'],values:[7.5,8.2,14.3,20.5,10.2,8.1],decimals:1,baseline:8,direction:'up',note:'Higher than previous baseline',tall:true},
 {id:'heart',name:'Resting heart rate',unit:'bpm',category:['Cardiovascular'],values:[62,65,78,88,70,64],decimals:0,baseline:64,direction:'up',note:'Above your usual resting pattern'},
 {id:'hrv',name:'HRV (RMSSD)',unit:'ms',category:['Cardiovascular','Nervous System'],values:[62,56,35,22,46,59],decimals:0,baseline:60,direction:'down',note:'Lower than previous baseline',tall:true},
 {id:'bp',name:'Blood pressure',unit:'mmHg',category:['Cardiovascular'],values:[116,119,128,138,121,117],decimals:0,baseline:118,direction:'up',note:'Above your usual baseline'},
 {id:'sleep',name:'Sleep duration',unit:'hrs',category:['Sleep','Nervous System'],values:[7.9,7.4,6.1,5.2,6.9,7.8],decimals:1,baseline:8,direction:'down',note:'Make room for a consistent bedtime',tall:true},
 {id:'quality',name:'Sleep quality',unit:'/ 10',category:['Sleep'],values:[9,8,6,4,7,9],decimals:0,baseline:8,direction:'down',note:'Below your usual sleep quality'},
 {id:'pss',name:'Perceived stress',unit:'/ 40',category:['Mood','Nervous System'],values:[9,13,23,31,18,11],decimals:0,baseline:12,direction:'up',note:'More demands than usual'},
 {id:'comfort',name:'Digestive comfort',unit:'/ 10',category:['Digestive'],values:[9,8,7,5,8,9],decimals:0,baseline:8,direction:'down',note:'Notice how meals and rest feel',tall:true},
 {id:'mood',name:'Mood check-in',unit:'/ 10',category:['Mood'],values:[8,7,5,4,6,8],decimals:0,baseline:8,direction:'down',note:'A little space for a check-in',tall:true}
];
export const diastolic=[76,78,83,89,79,76];
export const regions:Record<Region,{name:string;subtitle:string;metrics:string[]}>={
 head:{name:'Mental load',subtitle:'Rest, focus & perceived stress',metrics:['pss','sleep','quality']},
 shoulders:{name:'Muscle tension',subtitle:'A moment to soften your shoulders',metrics:['pss','sleep','heart']},
 heart:{name:'Cardiac context',subtitle:'Your rhythm & recovery',metrics:['heart','hrv','bp']},
 gut:{name:'Digestive wellbeing',subtitle:'Comfort, routine & nourishment',metrics:['comfort','glucose','sleep']}
};
export const records={
 'HRV Recordings':[{date:'12 Oct 2026',title:'Morning recovery check',detail:'5-minute resting session · 59 ms'},{date:'04 Sep 2026',title:'Quiet morning',detail:'5-minute resting session · 46 ms'},{date:'18 Aug 2026',title:'After a busy week',detail:'5-minute resting session · 22 ms'}],
 'Journal Entries':[{date:'10 Oct 2026',title:'Finding a steadier routine',detail:'A short walk and an earlier bedtime helped me unwind.'},{date:'08 Sep 2026',title:'Making room for breaks',detail:'I stepped away from my desk for ten minutes.'},{date:'22 Aug 2026',title:'A demanding week',detail:'I noticed tension and made time for a quiet check-in.'}]
};
export const historyOptions=['Stress History','Sleep History','Lifestyle'];
export const programs=['Everyday balance','Rest & recovery','Mindful movement'];
export const scanTypes=['Full body','Mental wellbeing','Cardiac context','Digestive wellbeing'];
export const recommendations:Record<Status,{title:string;text:string}[]>={
 Optimal:[{title:'Keep your rhythm',text:'Make a little room for the routines that feel good.'}],
 Elevated:[{title:'Take a breathing break',text:'Try a minute of slow, comfortable breathing.'},{title:'Protect your wind-down',text:'Choose a consistent time to step away from screens.'}],
 Critical:[{title:'Pause and reconnect',text:'Take a quiet break and check in with your counselor.'},{title:'One small step',text:'A gentle walk or a moment away from your desk may help you reset.'}]
};
export const statusColors:Record<Status,string>={Optimal:'#63bc95',Elevated:'#c9df70',Critical:'#f27b7d'};
export function metricStatus(metric:Metric,index:number):Status {
 const value=metric.values[index];const ratio=metric.direction==='up'?value/metric.baseline:metric.baseline/value;
 return ratio>1.65?'Critical':ratio>1.12?'Elevated':'Optimal';
}
export function filterMetrics(category:Category,search:string,overview:boolean){
 const query=search.trim().toLowerCase();
 return metrics.filter(m=>(overview||query!==''||m.category.includes(category))&&(!query||m.name.toLowerCase().includes(query)));
}
export function displayValue(metric:Metric,index:number){return metric.id==='bp'?`${metric.values[index]}/${diastolic[index]}`:metric.values[index].toFixed(metric.decimals);}
