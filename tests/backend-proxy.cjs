const assert = require('node:assert/strict');
const proxy = require('../serverless/backend_proxy');
async function call(route, body) {
  const res = {setHeader(){},status(n){this.code=n;return this;},json(body){this.body=body;return this;}};
  await proxy(route)({method:route==='predict'?'POST':'GET',body},res);
  return res;
}
(async()=>{
  delete process.env.INFERENCE_BACKEND_URL;delete process.env.INFERENCE_API_KEY;
  assert.equal((await call('health')).code,503);
  process.env.INFERENCE_BACKEND_URL='https://inference.example.test';
  process.env.INFERENCE_API_KEY='test-only-key';
  global.fetch=async(url,options)=>{
    assert.equal(options.headers.Authorization,'Bearer test-only-key');
    assert.equal(url.origin,'https://inference.example.test');
    return {ok:true,status:200,headers:new Headers({'content-type':'application/json'}),
      json:async()=>({model_version:'brfss2022_expanded',models:Object.fromEntries(['lr','rf','xgb','cnn','lstm','cnn-lstm'].map(n=>[n,{}]))})};
  };
  assert.equal((await call('predict',{})).code,200);
  assert.equal((await call('predict',{value:'x'.repeat(17000)})).code,413);
  global.fetch=async()=>{throw Error('offline')};
  assert.equal((await call('health')).code,502);
  process.env.INFERENCE_BACKEND_URL='http://insecure.example.test';
  assert.equal((await call('health')).code,502);
  console.log('PASS: proxy forwards authenticated requests and fails closed; no fabricated fallback.');
})().catch(e=>{console.error(e);process.exitCode=1});
