(() => {
  'use strict';
  const byId = id => document.getElementById(id);
  const form = byId('prediction-form');
  const fields = [
    ['heart_rate','Heart rate',55,110,1,75,'bpm'],
    ['systolic_bp','Systolic pressure',90,170,1,126,'mmHg'],
    ['diastolic_bp','Diastolic pressure',60,110,1,83,'mmHg'],
    ['sleep_duration','Sleep duration',4,10,.1,7.2,'hours'],
    ['quality_of_sleep','Sleep quality',1,10,1,7,'/ 10'],
    ['physical_activity','Physical activity',0,120,1,60,'min / day'],
    ['daily_steps','Daily steps',0,15000,100,7000,'steps']
  ];
  const defaults={age:35,gender:'Male',bmi_category:'Normal',sleep_disorder:'None',occupation:'Software Engineer'};
  const labels={age:'Age',gender:'Gender',bmi_category:'BMI category',sleep_disorder:'Sleep disorder',occupation:'Occupation'};
  const categories={gender:['Male','Female'],bmi_category:['Normal','Overweight','Obese'],sleep_disorder:['None','Insomnia','Sleep Apnea'],occupation:['Software Engineer','Doctor','Engineer','Lawyer','Manager','Nurse','Accountant','Salesperson','Scientist','Teacher','Sales Representative']};
  const models=[['Random Forest','CLASSICAL / ENSEMBLE'],['XGBoost','CLASSICAL / BOOSTING'],['Logistic Regression','CLASSICAL / LINEAR'],['1D CNN','NEURAL / CONVOLUTION'],['LSTM','NEURAL / RECURRENT'],['CNN-LSTM','NEURAL / HYBRID']];
  const presets={rest:[65,115,75,8,9,70,9000],day:[75,126,83,7.2,7,60,7000],busy:[85,140,90,5.5,4,25,3000]};
  const cards=new Map();let latest=null,revision=0,request=null;
  fields.forEach(([key,label,min,max,step,value,unit])=>{
    const row=document.createElement('div');row.className='signal-control';
    row.innerHTML=`<label for="signal-${key}">${label}<output id="signal-out-${key}" for="signal-${key}"></output></label><input id="signal-${key}" name="${key}" type="range" min="${min}" max="${max}" step="${step}" value="${value}">`;
    byId('prediction-sliders').append(row);
    const input=row.querySelector('input');
    const update=()=>{byId(`signal-out-${key}`).textContent='';byId(`signal-out-${key}`).append(document.createTextNode(Number(input.value).toLocaleString()));const small=document.createElement('small');small.textContent=unit;byId(`signal-out-${key}`).append(small);};
    input.addEventListener('input',()=>{update();document.querySelectorAll('[data-profile]').forEach(b=>b.setAttribute('aria-pressed','false'));invalidate();});update();
  });
  Object.keys(defaults).forEach(key=>{
    const label=document.createElement('label');label.textContent=labels[key];label.htmlFor=`signal-${key}`;
    const input=document.createElement(key==='age'?'input':'select');input.id=`signal-${key}`;input.name=key;input.required=true;
    if(key==='age'){input.type='number';input.min=18;input.max=100;input.step=1;}else categories[key].forEach(value=>{const option=document.createElement('option');option.value=value;option.textContent=value;input.append(option);});
    input.value=defaults[key];input.addEventListener('input',invalidate);label.append(input);byId('prediction-context').append(label);
  });
  models.forEach(([name,family],index)=>{
    const card=document.createElement('article');card.className='glass-panel model-card';card.setAttribute('aria-label',`${name} prediction`);
    card.innerHTML=`<div class="model-card-top"><span class="model-family">${family}</span><span class="model-number">0${index+1}</span></div><h3>${name}</h3><div class="model-verdict"><span class="model-label">Awaiting analysis</span><strong class="model-score">—</strong></div><div class="class-probabilities">${['Low','Moderate','High'].map(label=>`<div class="class-probability"><span>${label}</span><span class="probability-track"><i></i></span><output aria-label="${label} probability">—</output></div>`).join('')}</div><p class="probability-caption">Probability by stress category</p>`;
    cards.set(name,card);byId('prediction-cards').append(card);
  });
  function profile(){const data={};new FormData(form).forEach((value,key)=>{data[key]=key==='age'||fields.some(f=>f[0]===key)?Number(value):value;});return data;}
  function emptyCards(){cards.forEach(card=>{card.removeAttribute('data-level');card.querySelector('.model-label').textContent='Awaiting analysis';card.querySelector('.model-score').textContent='—';card.querySelectorAll('.probability-track i').forEach(bar=>bar.style.width='0%');card.querySelectorAll('output').forEach(output=>output.textContent='—');});}
  function invalidate(){revision++;latest=null;byId('download-prediction').disabled=true;byId('consensus-title').textContent='Ready to analyze.';byId('prediction-message').textContent='Profile updated. Run an analysis to refresh all six models.';byId('model-agreement').textContent='—';byId('agreement-fill').style.width='0%';emptyCards();}
  document.querySelectorAll('[data-profile]').forEach(button=>button.addEventListener('click',()=>{
    fields.forEach(([key,,,,,,unit],i)=>{const value=presets[button.dataset.profile][i];byId(`signal-${key}`).value=value;const out=byId(`signal-out-${key}`);out.textContent=Number(value).toLocaleString();const small=document.createElement('small');small.textContent=unit;out.append(small);});
    document.querySelectorAll('[data-profile]').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));invalidate();
  }));
  // Reveal invalid personal-context fields before native validity checking.
  form.addEventListener('invalid',event=>{if(event.target.closest('.personal-context'))event.target.closest('details').open=true;},true);
  form.addEventListener('submit',async event=>{
    event.preventDefault();const submitted=profile();
    if(submitted.systolic_bp<=submitted.diastolic_bp){byId('prediction-message').textContent='Systolic pressure must be higher than diastolic pressure. Please adjust your inputs.';byId('signal-systolic_bp').focus();return;}
    if(request)return;
    const submittedRevision=revision;latest=null;byId('download-prediction').disabled=true;emptyCards();
    byId('model-agreement').textContent='—';byId('agreement-fill').style.width='0%';
    byId('run-prediction').disabled=true;byId('run-prediction').textContent='Reading your signals…';byId('prediction-output').setAttribute('aria-busy','true');
    byId('consensus-title').textContent='Finding the patterns…';byId('prediction-message').textContent='Comparing your profile across all six trained models.';
    request=new AbortController();const timeout=setTimeout(()=>request?.abort(),30000);
    try{
      const response=await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(submitted),signal:request.signal});
      if(!response.headers.get('content-type')?.includes('application/json')){connection(false);throw new Error('The prediction service is unavailable. Open this page through the Python prediction server to run all six models.');}
      const data=await response.json();if(!response.ok||data.status!=='success')throw new Error(data.message||'The models could not complete this analysis.');
      if(submittedRevision!==revision)return;
      if(models.some(([name])=>!data.models?.[name]||!Array.isArray(data.models[name].probabilities)||data.models[name].probabilities.length!==3))throw new Error('The server returned incomplete model results. Please try again.');
      latest={created_at:new Date().toISOString(),profile:submitted,...data};
      byId('consensus-title').textContent=data.consensus.label;byId('model-agreement').textContent=`${data.consensus.agreement_rate}%`;byId('agreement-fill').style.width=`${data.consensus.agreement_rate}%`;
      byId('prediction-message').textContent='Majority vote across six models. Compare each prediction and its class probabilities below.';
      models.forEach(([name])=>{const result=data.models[name],card=cards.get(name);card.dataset.level=String(result.predicted_class);card.querySelector('.model-label').textContent=result.label;card.querySelector('.model-score').textContent=`${(result.confidence*100).toFixed(1)}%`;
        card.querySelectorAll('.class-probability').forEach((row,i)=>{row.querySelector('i').style.width=`${result.probabilities[i]*100}%`;row.querySelector('output').textContent=`${(result.probabilities[i]*100).toFixed(1)}%`;});});
      byId('download-prediction').disabled=false;connection(true);
    }catch(error){if(submittedRevision!==revision)return;byId('consensus-title').textContent='Let’s try that again.';byId('prediction-message').textContent=error.name==='AbortError'?'The analysis timed out. Please try again.':error instanceof TypeError?'Cannot reach the local prediction server. Check that it is running, then try again.':error.message;}
    finally{clearTimeout(timeout);request=null;byId('run-prediction').disabled=false;byId('run-prediction').innerHTML='Analyze all six models <span aria-hidden="true">↗</span>';byId('prediction-output').setAttribute('aria-busy','false');}
  });
  byId('download-prediction').addEventListener('click',()=>{if(!latest)return;const url=URL.createObjectURL(new Blob([JSON.stringify(latest,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='nerva-six-model-analysis.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
  function connection(ready){byId('connection').textContent=ready?'All six models connected':'Local server unavailable';byId('connection').dataset.state=ready?'ready':'error';}
  fetch('/api/health',{signal:AbortSignal.timeout(5000)}).then(r=>{if(!r.ok)throw Error();return r.json();}).then(data=>connection(data.models_loaded)).catch(()=>connection(false));
  fetch('benchmarks.json').then(r=>{if(!r.ok)throw Error();return r.json();}).then(rows=>rows.forEach(row=>{const tr=document.createElement('tr');[row.Model,`${(Number(row.Accuracy)*100).toFixed(2)}%`,Number(row['F1-Score (Macro)']).toFixed(4),Number(row['ROC-AUC (Macro)']).toFixed(4)].forEach(value=>{const td=document.createElement('td');td.textContent=value;tr.append(td);});byId('prediction-benchmarks').append(tr);})).catch(()=>{const tr=document.createElement('tr');const td=document.createElement('td');td.colSpan=4;td.textContent='Benchmark data is currently unavailable.';tr.append(td);byId('prediction-benchmarks').append(tr);});
  byId('about-predictions').addEventListener('click',()=>byId('about-dialog').close());
})();
