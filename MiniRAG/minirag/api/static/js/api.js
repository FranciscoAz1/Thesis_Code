// Clean, simplified SPA script implementing lastPage navigation and metadata query
(function(){
    const state = { apiKey: localStorage.getItem('apiKey') || '', files: [], currentPage: null, lastPage: null };

    function showToast(msg, dur = 2500){ const t=document.getElementById('toast'); if(!t) return; const inner=t.querySelector('div'); if(inner) inner.textContent=msg; t.classList.remove('hidden'); setTimeout(()=>t.classList.add('hidden'), dur); }
    function fetchWithAuth(url, opts={}){ const headers={...(opts.headers||{}), ...(state.apiKey? {Authorization:`Bearer ${state.apiKey}`}:{})}; return fetch(url,{...opts, headers}); }

    /* ---------------- Templates ---------------- */
    const pages = {
        'file-manager': () => `
            <div class="space-y-6">
                <h2 class="text-2xl font-bold text-gray-800">File Manager</h2>
                <div id="dropZone" class="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-gray-400 transition-colors cursor-pointer">
                    <p class="text-sm text-gray-500">Drag & drop or click to select files</p>
                    <input type="file" id="fileInput" multiple accept=".txt,.md,.doc,.docx,.pdf,.pptx" class="hidden" />
                </div>
                <div id="uploadProgress" class="hidden mt-2">
                    <div class="w-full bg-gray-200 rounded-full h-2.5"><div class="bg-blue-600 h-2.5 rounded-full" style="width:0%"></div></div>
                    <p class="text-xs text-gray-600 mt-1"><span id="uploadStatus">0/0</span></p>
                </div>
                <div class="flex gap-2">
                    <button id="uploadBtn" class="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700">Upload & Index</button>
                    <button id="rescanBtn" class="bg-gray-600 text-white px-4 py-2 rounded-lg hover:bg-gray-700">Rescan</button>
                </div>
                <div id="indexedFiles" class="space-y-2">
                    <h3 class="text-lg font-semibold text-gray-700">Indexed Files</h3>
                    <div id="indexedList" class="space-y-2"></div>
                </div>
            </div>` ,
        'query': () => `
            <div class="space-y-6">
                <h2 class="text-2xl font-bold text-gray-800">Query Database</h2>
                <div class="space-y-4">
                    <div>
                        <label class="block text-sm font-medium text-gray-700">Mode</label>
                        <select id="queryMode" class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500">
                            <option value="light">Light</option>
                            <option value="naive">Naive</option>
                            <option value="mini">Mini</option>
                            <option value="doc">Doc (enter doc_id)</option>
                            <option value="meta">Metadata</option>
                            <option value="bm25">BM25</option>
                        </select>
                        <button id="toggleContextOnly" type="button" class="mt-3 text-xs px-3 py-1 rounded bg-gray-200 hover:bg-gray-300 text-gray-800">Context Only: OFF</button>
                    </div>
                    <div id="standardQueryBlock">
                        <label class="block text-sm font-medium text-gray-700">Query</label>
                        <textarea id="queryInput" rows="4" class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"></textarea>
                    </div>
                    <div id="metadataQueryBlock" class="hidden">
                        <div class="flex items-center justify-between">
                            <label class="block text-sm font-medium text-gray-700">Metadata Filter</label>
                            <button id="addMetaFilter" class="text-xs px-2 py-1 bg-gray-200 rounded hover:bg-gray-300">Add</button>
                        </div>
                        <p class="text-xs text-gray-500 mb-1">All rows must match. Comma splits list values; lists match by overlap.</p>
                        <div id="metaFilterRows" class="space-y-2"></div>
                        <label class="inline-flex items-center text-xs mt-1"><input type="checkbox" id="showMetaJson" class="mr-1"> Show JSON</label>
                        <pre id="metaJsonPreview" class="hidden mt-2 p-2 bg-gray-100 text-xs rounded whitespace-pre-wrap"></pre>
                    </div>
                    <button id="queryBtn" class="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700">Send Query</button>
                    <div id="queryResult" class="mt-4 p-4 bg-white rounded-lg shadow text-sm"></div>
                </div>
            </div>` ,
        'knowledge-graph': () => `
            <div class="space-y-4 h-full flex flex-col">
                <h2 class="text-2xl font-bold text-gray-800">Knowledge Graph</h2>
                <div class="flex-1 border rounded-lg overflow-hidden shadow-sm relative">
                    <iframe id="kgFrame" src="/knowledge_graph.html" class="w-full h-full" frameborder="0"></iframe>
                </div>
                <p class="text-sm text-gray-500">Ensure knowledge_graph.html exists.</p>
            </div>` ,
        'file-info': () => `
            <div class="space-y-6">
                <h2 class="text-2xl font-bold text-gray-800">File Info</h2>
                <div id="fileInfo" class="p-4 bg-white rounded-lg shadow text-sm text-gray-700">Loading...</div>
                <button id="backFromFileInfo" class="text-blue-600 hover:underline text-sm">&larr; Back</button>
            </div>` ,
        'status': () => `
            <div class="space-y-6">
                <h2 class="text-2xl font-bold text-gray-800">System Status</h2>
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div class="p-6 bg-white rounded-lg shadow-sm"><h3 class="text-lg font-semibold mb-4">System Health</h3><div id="healthStatus"></div></div>
                    <div class="p-6 bg-white rounded-lg shadow-sm"><h3 class="text-lg font-semibold mb-4">Configuration</h3><div id="configStatus"></div></div>
                </div>
            </div>` ,
        'settings': () => `
            <div class="space-y-6">
                <h2 class="text-2xl font-bold text-gray-800">Settings</h2>
                <div class="max-w-xl"><div class="space-y-4">
                    <div><label class="block text-sm font-medium text-gray-700">API Key</label><input type="password" id="apiKeyInput" value="${state.apiKey}" class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500" /></div>
                    <button id="saveSettings" class="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700">Save Settings</button>
                </div></div>
            </div>`
    };

    /* ---------------- Handlers ---------------- */
    const handlers = {
        'file-manager': () => {
            const fileInput = document.getElementById('fileInput');
            const dropZone = document.getElementById('dropZone');
            const uploadBtn = document.getElementById('uploadBtn');
            const rescanBtn = document.getElementById('rescanBtn');
            const progress = document.getElementById('uploadProgress');
            const bar = progress.querySelector('.bg-blue-600');
            const statusEl = document.getElementById('uploadStatus');
            const indexedList = document.getElementById('indexedList');

            function updateIndexed(){
                fetchWithAuth('/health')
                    .then(r=>r.json())
                    .then(d=>{
                        indexedList.innerHTML = (d.indexed_files||[]).map(p=>{
                            const name = p.split(/[/\\]/).pop();
                            const enc = encodeURIComponent(name);
                            return `<div class="flex items-center justify-between bg-white p-3 rounded-lg shadow-sm"><span class="truncate">${name}</span><button class="text-sm text-blue-600 hover:underline ml-4" onclick="openFileInfo('${enc}')">Info</button></div>`;
                        }).join('');
                    }).catch(()=>{});
            }
            updateIndexed();

            dropZone.addEventListener('click', ()=> fileInput.click());
            dropZone.addEventListener('dragover', e=>{ e.preventDefault(); dropZone.classList.add('border-blue-500'); });
            dropZone.addEventListener('dragleave', ()=> dropZone.classList.remove('border-blue-500'));
            dropZone.addEventListener('drop', e=>{ e.preventDefault(); dropZone.classList.remove('border-blue-500'); state.files.push(...Array.from(e.dataTransfer.files)); showToast(`${state.files.length} file(s) selected`); });
            fileInput.addEventListener('change', ()=>{ state.files.push(...Array.from(fileInput.files)); fileInput.value=''; showToast(`${state.files.length} file(s) selected`); });

            uploadBtn.addEventListener('click', async ()=>{
                if(!state.files.length){ showToast('Select files first'); return; }
                progress.classList.remove('hidden');
                for(let i=0;i<state.files.length;i++){
                    const fd=new FormData(); fd.append('file', state.files[i]);
                        try { await fetchWithAuth('/documents/upload',{method:'POST', body:fd}); } catch(_) {}
                        bar.style.width = ((i+1)/state.files.length*100)+'%';
                        statusEl.textContent = `${i+1}/${state.files.length}`;
                }
                progress.classList.add('hidden');
                state.files=[];
                updateIndexed();
                showToast('Upload complete');
            });

            rescanBtn.addEventListener('click', async ()=>{ try { await fetchWithAuth('/documents/scan',{method:'POST'});} catch(_){} updateIndexed(); });
        },
        'query': () => {
            const queryMode = document.getElementById('queryMode');
            const queryBtn = document.getElementById('queryBtn');
            const queryResult = document.getElementById('queryResult');
            const standardBlock = document.getElementById('standardQueryBlock');
            const metadataBlock = document.getElementById('metadataQueryBlock');
            const metaRows = document.getElementById('metaFilterRows');
            const addMetaFilter = document.getElementById('addMetaFilter');
            const showMetaJson = document.getElementById('showMetaJson');
            const metaJsonPreview = document.getElementById('metaJsonPreview');
            const toggleContextOnly = document.getElementById('toggleContextOnly');
            let metadataKeys=[]; let keyValues={};
            let contextOnly = false;

            if (toggleContextOnly) {
                toggleContextOnly.addEventListener('click', () => {
                    contextOnly = !contextOnly;
                    toggleContextOnly.textContent = `Context Only: ${contextOnly ? 'ON' : 'OFF'}`;
                    toggleContextOnly.classList.toggle('bg-green-500', contextOnly);
                    toggleContextOnly.classList.toggle('text-white', contextOnly);
                    if (!contextOnly) {
                        toggleContextOnly.classList.remove('bg-green-500');
                    }
                });
            }

            async function fetchKeys(){ try { const r=await fetchWithAuth('/documents/metadata/keys'); if(!r.ok) return; const d=await r.json(); metadataKeys=(d.keys||[]).map(k=>k.key); keyValues=d.values||{}; } catch(_){} }
            function buildRow(k='',v=''){ const row=document.createElement('div'); row.className='flex gap-2 items-start'; row.innerHTML=`<div class="flex-1 relative"><input class="meta-k w-full border rounded px-2 py-1 text-xs" value="${k}" placeholder="key"><div class="suggestions-k absolute z-10 left-0 right-0 bg-white border rounded shadow max-h-40 overflow-auto hidden"></div></div><div class="flex-1 relative"><input class="meta-v w-full border rounded px-2 py-1 text-xs" value="${v}" placeholder="value (comma for list)"><div class="suggestions-v absolute z-10 left-0 right-0 bg-white border rounded shadow max-h-40 overflow-auto hidden"></div></div><button class="remove-meta-filter text-red-600 text-xs px-2 py-1">✕</button>`; return row; }
            function currentFilter(){ const o={}; metaRows.querySelectorAll('.flex.gap-2').forEach(r=>{ const k=r.querySelector('.meta-k').value.trim(); let v=r.querySelector('.meta-v').value.trim(); if(!k) return; if(v.includes(',')) v=v.split(',').map(s=>s.trim()).filter(Boolean); o[k]=v; }); metaJsonPreview.textContent=JSON.stringify(o,null,2); return o; }
            function renderSug(wrap, items, target){ if(!wrap)return; if(!items.length){wrap.classList.add('hidden'); return;} wrap.innerHTML=items.slice(0,50).map(i=>`<div class="px-2 py-1 hover:bg-blue-50 cursor-pointer text-xs">${i}</div>`).join(''); wrap.classList.remove('hidden'); wrap.querySelectorAll('div').forEach(d=>d.addEventListener('click',()=>{ target.value=d.textContent; currentFilter(); })); }

            metaRows.addEventListener('focusin', e=>{ if(e.target.classList.contains('meta-k')) renderSug(e.target.parentElement.querySelector('.suggestions-k'), metadataKeys, e.target); else if(e.target.classList.contains('meta-v')) { const key=e.target.closest('.flex.gap-2').querySelector('.meta-k').value.trim(); if(!key||!keyValues[key]) return; renderSug(e.target.parentElement.querySelector('.suggestions-v'), keyValues[key], e.target);} });
            metaRows.addEventListener('input', e=>{ if(e.target.classList.contains('meta-k')) renderSug(e.target.parentElement.querySelector('.suggestions-k'), metadataKeys.filter(x=>x.toLowerCase().includes(e.target.value.toLowerCase())), e.target); else if(e.target.classList.contains('meta-v')) { const key=e.target.closest('.flex.gap-2').querySelector('.meta-k').value.trim(); if(!key||!keyValues[key]) return; renderSug(e.target.parentElement.querySelector('.suggestions-v'), keyValues[key].filter(x=>x.toLowerCase().includes(e.target.value.toLowerCase())), e.target);} currentFilter(); });
            metaRows.addEventListener('click', e=>{ if(e.target.classList.contains('remove-meta-filter')) { e.target.parentElement.remove(); currentFilter(); }});
            metaRows.addEventListener('blur', ()=> setTimeout(()=> metaRows.querySelectorAll('.suggestions-k,.suggestions-v').forEach(s=>s.classList.add('hidden')), 120), true);
            addMetaFilter.addEventListener('click', ()=> metaRows.appendChild(buildRow()));
            showMetaJson.addEventListener('change', ()=> metaJsonPreview.classList.toggle('hidden', !showMetaJson.checked));
            queryMode.addEventListener('change', ()=>{ if(queryMode.value==='meta'){ standardBlock.classList.add('hidden'); metadataBlock.classList.remove('hidden'); if(!metaRows.querySelector('.flex.gap-2')) metaRows.appendChild(buildRow()); fetchKeys(); currentFilter(); } else { metadataBlock.classList.add('hidden'); standardBlock.classList.remove('hidden'); }});

            queryBtn.addEventListener('click', async ()=>{
                const mode=queryMode.value; let payload='';
                if(mode==='meta'){ const cf=currentFilter(); if(!Object.keys(cf).length){ showToast('Add a filter'); return;} payload=JSON.stringify(cf); }
                else { const qi=document.getElementById('queryInput'); if(!qi.value.trim()){ showToast('Enter a query'); return;} payload=qi.value.trim(); }
                queryBtn.disabled=true; const old=queryBtn.textContent; queryBtn.textContent='Processing...';
                try {
                    const r=await fetchWithAuth('/query',{method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({query:payload, mode, stream:false, only_need_context: contextOnly || mode==='meta'})});
                    const d=await r.json();
                    if(mode==='meta'){
                        let parsed; try { parsed=JSON.parse(d.response); } catch(_){}
                        if(parsed && parsed.matches){
                            if(!parsed.matches.length) queryResult.innerHTML='<div class="text-gray-600">No matches</div>';
                            else {
                                const items = parsed.matches.map(m=>{ const name=(m.file_path||m.doc_id||'').split(/[/\\]/).pop(); const enc=encodeURIComponent(name||''); return `<div class=\"flex items-center justify-between bg-white p-3 rounded-lg shadow-sm border\"><div class=\"min-w-0\"><div class=\"text-sm font-medium text-gray-800 truncate\">${name||'(no name)'} </div><div class=\"text-xs text-gray-500 mt-0.5\">doc_id: ${m.doc_id||''}</div></div><button class=\"text-sm text-blue-600 hover:underline ml-4\" onclick=\"openFileInfo('${enc}')\">Info</button></div>`; }).join('');
                                queryResult.innerHTML=`<h3 class=\"font-semibold text-gray-700 mb-2\">Matched Documents (${parsed.matches.length})</h3><div class=\"space-y-2\">${items}</div>`;
                            }
                        } else queryResult.innerHTML = (window.marked? marked.parse(d.response||''): (d.response||''));
                    } else queryResult.innerHTML = (window.marked? marked.parse(d.response||''): (d.response||''));
                } catch(_) { showToast('Query failed'); }
                finally { queryBtn.disabled=false; queryBtn.textContent=old; }
            });
        },
        'knowledge-graph': () => {
            const f=document.getElementById('kgFrame');
            function resize(){ if(!f) return; f.style.height = `${window.innerHeight - f.getBoundingClientRect().top - 24}px`; }
            window.addEventListener('resize', resize); resize();
        },
        'file-info': () => {
            const params=new URLSearchParams(location.hash.split('?')[1]||'');
            const fname=params.get('f');
            const back=document.getElementById('backFromFileInfo');
            if(back) back.onclick=()=> navigate(state.lastPage || 'file-manager');
            const box=document.getElementById('fileInfo');
            if(!fname){ box.textContent='No file selected'; return; }
            box.innerHTML='<div class="text-xs text-gray-500">Loading...</div>';
            (async()=>{
                try {
                    const r=await fetch(`/documents/info?filename=${encodeURIComponent(fname)}`);
                    if(!r.ok){ box.textContent='Error '+r.status; return; }
                    const d=await r.json();
                    box.innerHTML=`<div class="space-y-2">
                        <div><span class="font-semibold">Filename:</span> ${fname}</div>
                        <div><span class="font-semibold">Doc ID:</span> ${d.doc_id||'n/a'}</div>
                        <div><span class="font-semibold">Status:</span> ${d.status}</div>
                        <div><span class="font-semibold">Created:</span> ${d.created_at}</div>
                        <div><span class="font-semibold">Updated:</span> ${d.updated_at}</div>
                        <div><span class="font-semibold">Length:</span> ${d.content_length}</div>
                        <div><span class="font-semibold">Chunks:</span> ${d.chunks_count}</div>
                        <div class="mt-2"><span class="font-semibold">Metadata:</span><pre class="whitespace-pre-wrap bg-gray-100 p-2 rounded text-xs mt-1">${JSON.stringify(d.metadata||{},null,2)}</pre></div>
                        ${ (d.entities&&d.entities.length)? `<div class=\"mt-4\"><h3 class=\"font-semibold mb-1\">Entities</h3><div class=\"overflow-x-auto\"><table class=\"min-w-full text-xs border\"><thead><tr class=\"bg-gray-100\"><th class=\"px-2 py-1 border\">Name</th><th class=\"px-2 py-1 border\">Type</th><th class=\"px-2 py-1 border\">Description</th></tr></thead><tbody>${d.entities.map(e=>`<tr><td class=\\"px-2 py-1 border\\">${e.entity_name}</td><td class=\\"px-2 py-1 border\\">${e.entity_type||''}</td><td class=\\"px-2 py-1 border text-xs\\">${(e.description||'').slice(0,150)}</td></tr>`).join('')}</tbody></table></div></div>` : '<div class="mt-4 text-xs text-gray-500">No entities</div>' }
                        ${ (d.relationships&&d.relationships.length)? `<div class=\"mt-4\"><h3 class=\"font-semibold mb-1\">Relationships</h3><div class=\"overflow-x-auto\"><table class=\"min-w-full text-xs border\"><thead><tr class=\"bg-gray-100\"><th class=\"px-2 py-1 border\">Source</th><th class=\"px-2 py-1 border\">Target</th><th class=\"px-2 py-1 border\">Description</th><th class=\"px-2 py-1 border\">Keywords</th></tr></thead><tbody>${d.relationships.map(r=>`<tr><td class=\\"px-2 py-1 border\\">${r.src_id}</td><td class=\\"px-2 py-1 border\\">${r.tgt_id}</td><td class=\\"px-2 py-1 border text-xs\\">${(r.description||'').slice(0,150)}</td><td class=\\"px-2 py-1 border text-xs\\">${r.keywords||''}</td></tr>`).join('')}</tbody></table></div></div>` : '<div class="mt-4 text-xs text-gray-500">No relationships</div>' }
                    </div>`;
                } catch(_) { box.textContent='Failed to load'; }
            })();
        },
        'status': async () => {
            try { const r=await fetchWithAuth('/health'); const d=await r.json();
                document.getElementById('healthStatus').innerHTML = `
                    <div class="space-y-2">
                        <div class="flex items-center"><div class="w-3 h-3 rounded-full ${d.status==='healthy'?'bg-green-500':'bg-red-500'} mr-2"></div><span class="font-medium">${d.status}</span></div>
                        <div class="text-sm text-gray-600 space-y-1">
                            <div>Working Dir: ${d.working_directory}</div>
                            <div>Input Dir: ${d.input_directory}</div>
                            <div>Indexed Files: ${d.indexed_files_count}</div>
                        </div>
                    </div>`;
                document.getElementById('configStatus').innerHTML = Object.entries(d.configuration||{}).map(([k,v])=>`<div class="text-xs"><span class="font-medium">${k}:</span> <span class="text-gray-600">${v}</span></div>`).join('');
            } catch(_) { showToast('Status fetch failed'); }
        },
        'settings': () => {
            const save=document.getElementById('saveSettings');
            const apiKeyInput=document.getElementById('apiKeyInput');
            save.addEventListener('click', ()=>{ state.apiKey = apiKeyInput.value; localStorage.setItem('apiKey', state.apiKey); showToast('Saved'); });
        }
    };

    /* ---------------- Navigation ---------------- */
    function navigate(page, hash=''){
        if(page!==state.currentPage && state.currentPage) state.lastPage = state.currentPage;
        state.currentPage = page;
        location.hash = page + (hash?`?${hash}`:'');
        const content = document.getElementById('content');
        content.innerHTML = pages[page]();
        if(handlers[page]) handlers[page]();
    }
    window.navigate = navigate;
    window.openFileInfo = (fname)=> navigate('file-info', `f=${fname}`);

    // Attach nav item clicks (if sidebar already rendered)
    document.querySelectorAll('.nav-item').forEach(el=> el.addEventListener('click', e=>{ e.preventDefault(); navigate(el.dataset.page); }));

    // Initial page
    navigate('file-manager');
})();
