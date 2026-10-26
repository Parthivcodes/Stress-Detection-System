/* Local, dependency-free PDF export. Survey answers never leave the browser. */
(function (root) {
  'use strict';
  const ascii = value => String(value ?? '').normalize('NFKD').replace(/[\u0300-\u036f]/g, '')
    .replace(/[\u2010-\u2015\u2212]/g, '-').replace(/[\u2018\u2019]/g, "'").replace(/[\u201c\u201d]/g, '"')
    .replace(/\u2265/g, '>=').replace(/\u2264/g, '<=').replace(/\u00b2/g, '2').replace(/[^\x20-\x7e]/g, ' ');
  const escape = value => ascii(value).replace(/([\\()])/g, '\\$1');
  const pct = value => Number.isFinite(Number(value)) ? `${(100 * Number(value)).toFixed(1)}%` : 'Unavailable';
  const labels = ['Rarely / never', 'Sometimes', 'Usually / always'];
  function create(report, options = {}) {
    if (!report?.consensus || !report.models || !report.profile) throw new Error('Run an analysis before downloading its PDF.');
    const names = options.modelOrder || Object.keys(report.models);
    if (names.length !== 6 || names.some(n => !report.models[n])) throw new Error('The report requires all six model results.');
    const pages = []; let commands = []; const height = 841.89;
    const ink = '0.13 0.15 0.15', muted = '0.37 0.40 0.40', gold = '0.57 0.43 0.26';
    const text = (value, x, top, size = 10, font = 'F1', color = ink) => {
      commands.push(`BT /${font} ${size} Tf ${color} rg 1 0 0 1 ${x} ${(height-top).toFixed(2)} Tm (${escape(value)}) Tj ET`);
    };
    const rect = (x, top, w, h, color) => commands.push(`${color} rg ${x} ${(height-top-h).toFixed(2)} ${w} ${h} re f`);
    const line = (top) => rect(44, top, 507, 0.7, '0.84 0.85 0.83');
    // Conservative glyph-width upper bound for built-in Helvetica, including wide capitals.
    const width = (value, size) => [...ascii(value)].reduce((sum,c) => sum + (/[MW@%]/.test(c) ? .95 : /[ilI.,:;!'| ]/.test(c) ? .30 : /[A-Z]/.test(c) ? .73 : .57), 0) * size;
    const wrap = (value, maxWidth, size) => {
      const result = []; let current = '';
      for (const word of ascii(value).split(/\s+/)) {
        if (current && width(`${current} ${word}`,size)>maxWidth) { result.push(current); current=''; }
        if (width(word,size)>maxWidth) {
          for (const char of word) { if (width(current+char,size)>maxWidth) { result.push(current); current=''; } current+=char; }
        } else current += (current?' ':'')+word;
      }
      if(current)result.push(current); return result;
    };
    const paragraph = (value, top, size=10, color=muted, maxWidth=507, x=44) => {
      const lines=wrap(value,maxWidth,size); lines.forEach((s,i)=>text(s,x,top+i*(size+5),size,'F1',color)); return top+lines.length*(size+5);
    };
    const page = (number, title) => {
      commands=[]; rect(0,0,595.28,7,gold);text('NERVA',44,44,19,'F2');text('HUMAN SIGNALS LAB',147,43,8,'F1',muted);
      text(title,44,87,25,'F2'); line(105);
      text('Academic model output - not a clinical diagnosis',44,806,8,'F1',muted);text(`Page ${number} of 2`,490,806,8,'F1',muted);
    };
    const finish = () => pages.push(commands.join('\n'));
    const date = new Date(report.created_at); const dateText = Number.isNaN(date.getTime()) ? 'Unavailable' : date.toLocaleString('en-GB',{dateStyle:'medium',timeStyle:'short'});
    page(1,'Your stress-frequency analysis');
    text(`Prepared ${dateText}`,44,127,9,'F1',muted);
    rect(44,147,507,89,'0.96 0.95 0.92');text('COMBINED MODEL RESULT',58,168,8,'F2',gold);
    text(report.consensus.label || labels[report.consensus.predicted_class] || 'Unavailable',58,195,21,'F2');
    text('Frequency of reported stress; this is not a severity diagnosis.',58,220,9,'F1',muted);
    const soft=report.consensus.method==='weighted_raw_probabilities_with_high_cutoff';
    let top=paragraph(soft ? 'The combined result averages selected models and applies a validation-selected decision threshold. Individual models can disagree.' : 'The combined result uses the decision rule configured by the prediction service. Individual models can disagree.',257);
    const cp=report.consensus.probabilities;
    if(Array.isArray(cp)&&cp.length===3){
      labels.forEach((label,i)=>{text(label,44,top+15,9);rect(192,top+6,260,10,'0.91 0.92 0.90');rect(192,top+6,260*Math.max(0,Math.min(1,Number(cp[i])||0)),10,gold);text(pct(cp[i]),474,top+15,10,'F2');top+=27;});
    }
    top+=24;text('HOW THE SIX MODELS COMPARE',44,top,10,'F2');top+=24;
    text('Model / predicted stress frequency',44,top,9,'F2');text('Rarely',341,top,8,'F2');text('Sometimes',397,top,8,'F2');text('Usually',475,top,8,'F2');top+=10;line(top);
    names.forEach((name,i)=>{
      top+=18;const model=report.models[name];text(name,44,top,11,'F2');text(model.label||labels[model.predicted_class]||'Unavailable',44,top+15,9,'F1',muted);
      [343,407,477].forEach((x,j)=>text(pct(model.probabilities?.[j]),x,top+7,9));top+=28;line(top);
    });
    top+=20;paragraph('The three columns show each model\'s estimated probabilities. They are uncalibrated estimates, not your measured likelihood of a medical condition. Individual model adjustments and the combined decision rule may differ.',top,9);
    finish();
    page(2,'The profile behind this result');
    paragraph('These are the answers submitted for this analysis. Values below are written in everyday language so this report can be read without the application.',128,10);
    top=177;
    const numeric = new Map((options.fields||[]).map(f=>[f[0],f]));
    Object.entries(report.profile).forEach(([key,value],i)=>{
      const f=numeric.get(key);let label=options.labels?.[key]||f?.[1]||key.replace(/_/g,' '); let display;
      if(options.categories?.[key]) display=options.categories[key][Number(value)-1] || `Unrecognized value (${value})`;
      else if(key==='age') display=Number(value)===80?'80 years or older':`${value} years`;
      else if(key==='sleep_duration') display=`${value} hours per night`;
      else if(key==='unhealthy_days') display=`${value} days in the past 30 days`;
      else if(key==='bmi') display=`${Number(value).toFixed(2)} kg/m2`;
      else display=String(value);
      const ll=wrap(label,290,9),vl=wrap(display,180,9);const h=Math.max(ll.length,vl.length)*12+8;
      if(i%2===0)rect(44,top-12,507,h,'0.96 0.97 0.96');
      ll.forEach((s,j)=>text(s,53,top+j*12,9));vl.forEach((s,j)=>text(s,360,top+j*12,9,'F2'));top+=h;
    });
    top+=24;text('WHAT THIS REPORT CAN TELL YOU',44,top,10,'F2');top+=19;
    top=paragraph('The models look for patterns associated with concurrent self-reported stress frequency. They do not establish a diagnosis, explain the cause of stress, or predict future health.',top,9);
    top+=9;paragraph(`Model data: ${report.dataset||'Not supplied'}${report.model_version?` (${report.model_version})`:''}. The website\'s displayed class probabilities and the submitted profile are preserved in this report.`,top,9);
    finish();
    const objects=[];const add=body=>(objects.push(body),objects.length);
    add('<< /Type /Catalog /Pages 2 0 R >>'); add('');
    add('<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>');add('<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>');
    const kids=[];
    pages.forEach(stream=>{const pageId=add('');const streamId=add(`<< /Length ${stream.length} >>\nstream\n${stream}\nendstream`);objects[pageId-1]=`<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595.28 841.89] /Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> /Contents ${streamId} 0 R >>`;kids.push(`${pageId} 0 R`);});
    objects[1]=`<< /Type /Pages /Kids [${kids.join(' ')}] /Count ${pages.length} >>`;
    let pdf='%PDF-1.4\n';const offsets=[0];objects.forEach((body,i)=>{offsets.push(pdf.length);pdf+=`${i+1} 0 obj\n${body}\nendobj\n`;});
    const xref=pdf.length;pdf+=`xref\n0 ${objects.length+1}\n0000000000 65535 f \n`;offsets.slice(1).forEach(o=>pdf+=`${String(o).padStart(10,'0')} 00000 n \n`);
    pdf+=`trailer\n<< /Size ${objects.length+1} /Root 1 0 R >>\nstartxref\n${xref}\n%%EOF\n`;
    return new TextEncoder().encode(pdf);
  }
  root.NervaAnalysisPDF={create};
})(typeof window !== 'undefined' ? window : globalThis);
