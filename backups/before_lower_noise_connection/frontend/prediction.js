(async () => {
  'use strict';
  const byId = id => document.getElementById(id);
  const form = byId('prediction-form');
  let schema = null, simulation = false;
  try {
    byId('prediction-message').textContent='Connecting to the prediction service…';
    const healthResponse=await fetch('/api/health',{signal:AbortSignal.timeout(30000)});
    const health=healthResponse.ok?await healthResponse.json():null;
    if(!health?.models_loaded || !['brfss2022','model_training_simulation'].includes(health.dataset))throw new Error('Models unavailable');
    simulation=health.model_version==='model_training_simulation';
    if(health?.model_version==='brfss2022_expanded'||simulation){
      const response=await fetch('/api/schema',{signal:AbortSignal.timeout(30000)});
      if(!response.ok)throw new Error('Expanded input schema unavailable');
      schema=await response.json();
    }
  }catch(error){
    byId('prediction-message').textContent='The prediction service is unavailable or starting up. Wait a moment, then refresh to reconnect.';
    byId('run-prediction').disabled=true;
    return;
  }
  const expanded=schema?.model_version==='brfss2022_expanded'||simulation;
  const highRecallHeader=document.createElement('th');highRecallHeader.textContent='High-frequency recall';document.querySelector('.evidence-table thead tr').append(highRecallHeader);
  const fields = [
    ['age','Age (80 means 80+)',18,80,1,35,'years'],
    ['sleep_duration','Average sleep',1,24,1,7,'hours'],
    ['unhealthy_days','Physically unhealthy days',0,30,1,0,'past 30 days'],
    ['bmi','Body mass index',1,99.98,.01,24,'kg / m²']
  ];
  const defaults={sex:1,exercise:1,general_health:2,marital:1,education:6,employment:1,income:7,support:2};
  const labels={sex:'Survey-recorded sex',exercise:'Exercise in past month',general_health:'General health',marital:'Marital status',education:'Education',employment:'Employment',income:'Annual household income (USD)',support:'Emotional / social support'};
  const categories={sex:['Male','Female'],exercise:['Yes','No'],general_health:['Excellent','Very good','Good','Fair','Poor'],marital:['Married','Divorced','Widowed','Separated','Never married','Unmarried couple'],education:['No school / kindergarten','Grades 1–8','Grades 9–11','High school / GED','Some college / technical','College 4+ years'],employment:['Employed for wages','Self-employed','Out of work ≥1 year','Out of work <1 year','Homemaker','Student','Retired','Unable to work'],income:['Under $10,000','$10,000–14,999','$15,000–19,999','$20,000–24,999','$25,000–34,999','$35,000–49,999','$50,000–74,999','$75,000–99,999','$100,000–149,999','$150,000–199,999','$200,000+'],support:['Always','Usually','Sometimes','Rarely','Never']};
  const models=[['Random Forest','CLASSICAL / ENSEMBLE'],['XGBoost','CLASSICAL / BOOSTING'],['Logistic Regression','CLASSICAL / LINEAR'],['1D CNN','NEURAL / CONVOLUTION'],['LSTM','NEURAL / RECURRENT'],['CNN-LSTM','NEURAL / HYBRID']];
  const presets={rest:[35,8,0,24],day:[35,7,0,24],busy:[35,5,15,30]};
  if(expanded){
    Object.assign(defaults,{isolation:5,job_loss:2,food_insecurity:5,bill_difficulty:2,utility_threat:2,transport_barrier:2,medical_cost:2});
    Object.assign(labels,{isolation:'How often do you feel socially isolated?',job_loss:'Lost work / reduced hours (past year)',food_insecurity:'Food ran out, no money for more (past year)',bill_difficulty:'Unable to pay housing / utility bills (past year)',utility_threat:'Utility shutoff threatened (past year)',transport_barrier:'Transport prevented necessary activities (past year)',medical_cost:'Could not afford a needed doctor visit (past year)'});
    Object.assign(categories,{isolation:['Always','Usually','Sometimes','Rarely','Never'],food_insecurity:['Always','Usually','Sometimes','Rarely','Never'],job_loss:['Yes','No'],bill_difficulty:['Yes','No'],utility_threat:['Yes','No'],transport_barrier:['Yes','No'],medical_cost:['Yes','No']});
    form.querySelector('.subtle-count').textContent='19 signals';
    document.querySelector('.intro-copy').innerHTML=document.querySelector('.intro-copy').innerHTML.replace('Twelve','Nineteen');
    form.querySelector('.input-caption').textContent='Illustrative examples, not real respondent profiles. All nineteen survey answers are required. Income is in USD. Processing stays local.';
    const method=document.querySelectorAll('.prediction-method article p');
    method[0].textContent='Nineteen survey inputs cover health, sleep, social support, isolation and financial or access difficulties.';
    method[2].textContent=simulation?`${schema.records.toLocaleString()} generated profiles, with ${schema.training_records.toLocaleString()} in the training partition. This educational simulation does not establish prediction accuracy for real people.`:`${schema.respondents.toLocaleString()} eligible real BRFSS respondents; ${schema.training_respondents.toLocaleString()} used for training. Predicts concurrent reported stress frequency, not a diagnosis or future stress.`;
    document.querySelector('.evidence-heading .subtle-count').textContent=simulation?`SIMULATION TEST · ${schema.test_records.toLocaleString()} PROFILES`:`NEW HOLDOUT · ${schema.test_respondents.toLocaleString()} RESPONDENTS`;
    document.querySelector('.prediction-evidence>p').textContent=simulation?'Accuracy below is measured on held-out generated profiles from model_training_data.csv. Individual results use the largest raw class probability. These scores measure learning the simulation rules and are not evidence of accuracy on real people.':'Previously unused respondents and input profiles. Settings were selected on validation data with a high-stress recall constraint. Every promoted model exceeded both prior versions in observed high-stress recall and accuracy on this same holdout. Middle-category recall declined. This is not a guarantee for future people or external clinical validation.';
    const about=document.querySelector('#about-dialog p');about.textContent=about.textContent.replace('twelve profile inputs','nineteen profile inputs');
    const baseline=document.createElement('p');
    baseline.textContent=simulation?`Always predicting rarely / never: ${(100*schema.majority_accuracy).toFixed(2)}% accuracy on this simulation test; 0% high-stress and sometimes recall.`:'Always predicting rarely / never: 65.78% accuracy on this holdout; 0% high-stress recall.';
    document.querySelector('.prediction-evidence').insertBefore(baseline,document.querySelector('.evidence-table'));
    if(schema.ensemble?.method==='weighted_raw_probabilities_with_high_cutoff'){
      method[1].textContent='Six individual model results remain available. The consensus uses a validation-selected weighted blend of raw probabilities and one high-stress cutoff, not majority voting.';
      const m=schema.ensemble.test_metrics;
      const evidence=document.createElement('p');
      evidence.textContent=`Frozen soft ensemble: accuracy ${(100*m.accuracy).toFixed(2)}%; macro-F1 ${m.macro_f1.toFixed(4)}; QWK ${m.qwk.toFixed(4)}; sometimes recall ${(100*m.recall[1]).toFixed(2)}%; high recall ${(100*m.recall[2]).toFixed(2)}%; macro AUC ${m.macro_auc_ovr.toFixed(4)}. Reused internal holdout, not external validation.`;
      document.querySelector('.prediction-evidence').insertBefore(evidence,document.querySelector('.evidence-table'));
      document.querySelector('.agreement-caption').textContent='Individual matches, not vote-based selection';
    }
  }
  if(simulation){
    const badge=document.createElement('p');badge.className='micro-label';badge.textContent='EDUCATIONAL SIMULATION / GENERATED TRAINING DATA';byId('prediction-heading').after(badge);
    document.querySelector('.intro-copy').textContent='Nineteen inputs, six model families. Explore a separately trained educational simulation using generated profiles.';
    form.querySelector('.input-caption').textContent='Generated-data demonstration. All nineteen coded profile inputs are required. Results are not validated for real people.';
    const method=document.querySelectorAll('.prediction-method article p');method[0].textContent='The same nineteen coded profile inputs cover health, sleep, social context and access difficulties.';method[1].textContent='Each model selects its largest raw class probability. The combined display averages the six probability vectors equally. Neural families average two fixed training seeds.';
    document.querySelector('#about-dialog p').textContent='This preview demonstrates six models trained using 10,000 programmatically generated profiles with nineteen coded inputs. Accuracy is measured on held-out generated profiles; real-person accuracy has not been established.';
    document.querySelector('.agreement-caption').textContent='Individual matches with the probability mean';
  }
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
    categories[key].forEach((value,index)=>{const option=document.createElement('option');option.value=index+1;option.textContent=value;input.append(option);});
    input.value=defaults[key];input.addEventListener('input',invalidate);label.append(input);byId('prediction-context').append(label);
  });
  models.forEach(([name,family],index)=>{
    const card=document.createElement('article');card.className='glass-panel model-card';card.setAttribute('aria-label',`${name} prediction`);
    card.innerHTML=`<div class="model-card-top"><span class="model-family">${family}</span><span class="model-number">0${index+1}</span></div><h3>${name}</h3><div class="model-verdict"><span class="model-label">Awaiting analysis</span><strong class="model-score">—</strong></div><div class="class-probabilities">${['Rarely/never','Sometimes','Usually/always'].map(label=>`<div class="class-probability"><span>${label}</span><span class="probability-track"><i></i></span><output aria-label="${label} probability">—</output></div>`).join('')}</div><p class="probability-caption">Estimated stress frequency</p>`;
    cards.set(name,card);byId('prediction-cards').append(card);
  });
  function profile(){const data={};new FormData(form).forEach((value,key)=>{data[key]=Number(value);});return data;}
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
      if(simulation?data.data_origin!=='synthetic_simulation':data.dataset!=='BRFSS 2022')throw new Error('This server is using a different model set. Refresh the page to reconnect.');
      if(submittedRevision!==revision)return;
      if(models.some(([name])=>!data.models?.[name]||!Array.isArray(data.models[name].probabilities)||data.models[name].probabilities.length!==3))throw new Error('The server returned incomplete model results. Please try again.');
      latest={created_at:new Date().toISOString(),profile:submitted,...data};
      byId('consensus-title').textContent=data.consensus.label;byId('model-agreement').textContent=`${data.consensus.agreement_rate}%`;byId('agreement-fill').style.width=`${data.consensus.agreement_rate}%`;
      byId('prediction-message').textContent=simulation?'Educational simulation: the combined result averages six raw probability vectors equally. Each card selects its largest class probability. Results are not validated for real people.':'Majority vote across six models. Compare each prediction and its class probabilities below.';
      if(data.consensus.method==='weighted_raw_probabilities_with_high_cutoff'){
        const mix=Object.entries(data.consensus.weights).map(([name,w])=>`${name} ${(100*w).toFixed(0)}%`).join(' + ');
        const probabilities=data.consensus.probabilities.map((p,i)=>`${['Rarely/never','Sometimes','Usually/always'][i]} ${(100*p).toFixed(2)}%`).join(' | ');
        byId('prediction-message').textContent=`Soft ensemble: ${mix}. ${probabilities}. Decision: usually/always when its probability is at least ${(100*data.consensus.high_threshold).toFixed(4)}%; otherwise choose the larger rarely/sometimes probability. The six cards retain their individual class adjustments.`;
      }
      models.forEach(([name])=>{const result=data.models[name],card=cards.get(name);card.dataset.level=String(result.predicted_class);card.querySelector('.model-label').textContent=result.label;card.querySelector('.model-score').textContent=`${(result.confidence*100).toFixed(1)}%`;
        card.querySelectorAll('.class-probability').forEach((row,i)=>{row.querySelector('i').style.width=`${result.probabilities[i]*100}%`;row.querySelector('output').textContent=`${(result.probabilities[i]*100).toFixed(1)}%`;});});
      byId('download-prediction').disabled=false;connection(true);
    }catch(error){if(submittedRevision!==revision)return;byId('consensus-title').textContent='Let’s try that again.';byId('prediction-message').textContent=error.name==='AbortError'?'The analysis timed out. Please try again.':error instanceof TypeError?'Cannot reach the local prediction server. Check that it is running, then try again.':error.message;}
    finally{clearTimeout(timeout);request=null;byId('run-prediction').disabled=false;byId('run-prediction').innerHTML='Analyze all six models <span aria-hidden="true">↗</span>';byId('prediction-output').setAttribute('aria-busy','false');}
  });
  byId('download-prediction').innerHTML='Download analysis PDF <span aria-hidden="true">↓</span>';
  byId('download-prediction').addEventListener('click',()=>{
    if(!latest)return;
    try{
      if(!window.NervaAnalysisPDF)throw new Error('The PDF exporter could not load. Refresh this page and try again.');
      const pdf=window.NervaAnalysisPDF.create(latest,{fields,labels,categories,modelOrder:models.map(([name])=>name)});
      const url=URL.createObjectURL(new Blob([pdf],{type:'application/pdf'}));const a=document.createElement('a');a.href=url;a.download=`nerva-analysis-${latest.created_at.slice(0,10)}.pdf`;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),60000);
    }catch(error){byId('prediction-message').textContent=error.message||'The PDF could not be created. Please try again.';}
  });
  function connection(ready){byId('connection').textContent=ready?(simulation?'Simulation: six models connected':'All six models connected'):'Local server unavailable';byId('connection').dataset.state=ready?'ready':'error';}
  fetch('/api/health',{signal:AbortSignal.timeout(5000)}).then(r=>{if(!r.ok)throw Error();return r.json();}).then(data=>connection(data.models_loaded&&data.dataset===(simulation?'model_training_simulation':'brfss2022'))).catch(()=>connection(false));
  fetch('/api/benchmarks').then(r=>{if(!r.ok)throw Error();return r.json();}).then(rows=>rows.forEach(row=>{const tr=document.createElement('tr');[row.Model,`${(Number(row.Accuracy)*100).toFixed(2)}%`,Number(row['F1-Score (Macro)']).toFixed(4),Number(row['ROC-AUC (Macro)']).toFixed(4),`${(100*(row['Recall by class']?.[2]??(row.confusion_matrix?.[2]?.[2]/row.confusion_matrix?.[2]?.reduce((a,b)=>a+b,0)))).toFixed(2)}%`].forEach(value=>{const td=document.createElement('td');td.textContent=value;tr.append(td);});byId('prediction-benchmarks').append(tr);})).catch(()=>{const tr=document.createElement('tr');const td=document.createElement('td');td.colSpan=5;td.textContent='Benchmark data is currently unavailable.';tr.append(td);byId('prediction-benchmarks').append(tr);});
  byId('about-predictions').addEventListener('click',()=>byId('about-dialog').close());
})();
