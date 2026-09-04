let decisionsData = {};

async function fetchDecisions() {
    try {
        const response = await fetch('/api/decisions');
        const data = await response.json();
        
        const tbody = document.getElementById('decision-table-body');
        tbody.innerHTML = '';
        
        let totalRevenue = 0;
        let totalOverrides = 0;
        let totalSuccess = 0;
        let feedbackCount = 0;
        
        data.forEach(item => {
            decisionsData[item.id] = item; 
            
            // Tally outcomes and revenue
            if (item.outcome_status === "RECOVERED") {
                totalSuccess++;
                totalRevenue += item.recovered_amount || 0;
            }
            if (item.action_overridden) totalOverrides++;
            if (item.feedback_queue && item.feedback_queue !== "PENDING") feedbackCount++;
            
            const time = item.timestamp ? new Date(item.timestamp).toLocaleTimeString() : "Just now";
            const aiAction = toTitleCase(item.original_action || item.recommended_action);
            const finalAction = toTitleCase(item.recommended_action);
            const reason = item.failure_reason || "No reason provided";
            
            // Policy badge logic
            let policyBadge = `<span style="color: #38a169; font-weight: 600;">✅ Passed</span>`;
            if (item.action_overridden) {
                policyBadge = `<span style="color: #e53e3e; font-weight: 600;" title="Guardrail Triggered">🛑 Override</span>`;
            }

            // Outcome badge logic
            let outcomeBadge = `<span style="color: #718096; font-weight: bold;">Pending</span>`;
            if (item.outcome_status === "RECOVERED") {
                outcomeBadge = `<span style="color: #38a169; font-weight: bold;">✔ Recovered (₹${(item.recovered_amount || 0).toLocaleString('en-IN', {minimumFractionDigits: 2})})</span>`;
            } else if (item.outcome_status === "FAILED") {
                outcomeBadge = `<span style="color: #e53e3e; font-weight: bold;">✖ Failed</span>`;
            } else if (item.outcome_status === "NOT_ATTEMPTED") {
                outcomeBadge = `<span style="color: #a0aec0; font-weight: bold;">⊘ Suppressed</span>`;
            }

            const row = `
                <tr class="clickable" onclick="openModal('${item.id}')">
                    <td><strong>${item.transaction_id || 'TXN'}</strong><br><small>${time}</small></td>
                    <td>₹${(item.amount || 0).toLocaleString('en-IN', {minimumFractionDigits: 2})}</td>
                    <td>${reason}</td>
                    <td>${aiAction}</td>
                    <td>${policyBadge}</td>
                    <td><span class="badge ${item.recommended_action}">${finalAction}</span></td>
                    <td>${outcomeBadge}</td>
                </tr>
            `;
            tbody.innerHTML += row;
        });

        // Update Expanded KPIs
        document.getElementById('kpi-total').innerText = data.length;
        document.getElementById('kpi-success').innerText = totalSuccess;
        
        const recoveryRate = data.length > 0 ? ((totalSuccess / data.length) * 100).toFixed(1) : 0;
        document.getElementById('kpi-recovery-rate').innerText = recoveryRate + '%';
        
        document.getElementById('kpi-revenue').innerText = '₹' + totalRevenue.toLocaleString('en-IN', {maximumFractionDigits: 0});
        document.getElementById('kpi-feedback').innerText = `${feedbackCount} queued`;
        
    } catch (error) {
        console.error("Error fetching decisions:", error);
    }
}

function toTitleCase(str) {
    if (!str) return "Unknown Action";
    return str.replace('_', ' ').replace(
        /\w\S*/g,
        function(txt) { return txt.charAt(0).toUpperCase() + txt.substr(1).toLowerCase(); }
    );
}

function safeProb(val) {
    const num = Number(val);
    return !isNaN(num) ? (num * 100).toFixed(1) : "0.0";
}

function openModal(id) {
    const item = decisionsData[id];
    if (!item) return;

    const actionName = toTitleCase(item.recommended_action);

    // 1. Header & Dynamic Summary (AI vs Guardrail Flowchart)
    document.getElementById('modal-txn-id').innerText = `Decision Intel: ${item.transaction_id || id}`;
    
    let summaryHTML = "";
    if (item.action_overridden) {
        const origData = item.alternative_actions ? item.alternative_actions.find(a => a.action === item.original_action) : null;
        const origProb = origData ? safeProb(origData.probability) : "N/A";
        const origRev = origData ? origData.expected_revenue.toLocaleString('en-IN', {minimumFractionDigits: 2}) : "N/A";
        const flagText = item.guardrail_flags && item.guardrail_flags.length > 0 ? item.guardrail_flags[0] : "Hard constraint triggered";

        summaryHTML = `
            <div style="margin-bottom: 10px; font-family: sans-serif; font-size: 0.95rem;">
                <div style="background: #f8fafc; padding: 12px; border-radius: 6px; border: 1px solid #e2e8f0; margin-bottom: 6px;">
                    <strong>🤖 AI PROPOSAL</strong><br>
                    <span style="font-size: 1.05rem; color: #2d3748;">${toTitleCase(item.original_action)}</span><br>
                    <span style="color: #718096; font-size: 0.9rem;">${origProb}% recovery probability · ₹${origRev} expected recovery</span>
                </div>
                
                <div style="text-align: center; color: #cbd5e0; font-weight: bold; margin: 4px 0;">↓</div>
                
                <div style="background: #fff5f5; padding: 12px; border-radius: 6px; border-left: 4px solid #e53e3e; margin-bottom: 6px;">
                    <strong style="color: #c53030;">🛑 POLICY CHECK</strong><br>
                    <span style="font-size: 0.9rem; color: #4a5568;">${flagText}</span><br>
                    <strong style="font-size: 0.85rem; color: #e53e3e; letter-spacing: 0.5px;">OUTREACH SUPPRESSED</strong>
                </div>
                
                <div style="text-align: center; color: #cbd5e0; font-weight: bold; margin: 4px 0;">↓</div>
                
                <div style="background: #f0fff4; padding: 12px; border-radius: 6px; border-left: 4px solid #38a169;">
                    <strong style="color: #276749;">🔴 FINAL ACTION (ENFORCED)</strong><br>
                    <span style="font-size: 1.05rem; color: #2d3748;">${actionName}</span><br>
                    <span style="color: #718096; font-size: 0.9rem;">${safeProb(item.predicted_recovery_probability || item.predicted_probability)}% · ₹${(item.expected_revenue || 0).toLocaleString('en-IN', {minimumFractionDigits: 2})}</span>
                </div>
            </div>
        `;
    } else {
        summaryHTML = `
            <div style="margin-bottom: 10px; font-family: sans-serif; font-size: 0.95rem;">
                <div style="background: #f0fff4; padding: 12px; border-radius: 6px; border-left: 4px solid #38a169; margin-bottom: 6px;">
                    <strong style="color: #276749;">🤖 AI PROPOSAL & POLICY ALIGNED</strong><br>
                    <span style="font-size: 1.05rem; color: #2d3748;">${actionName}</span><br>
                    <span style="color: #718096; font-size: 0.9rem;">${safeProb(item.predicted_recovery_probability || item.predicted_probability)}% recovery probability · ₹${(item.expected_revenue || 0).toLocaleString('en-IN', {minimumFractionDigits: 2})} expected recovery</span>
                </div>
                <div style="background: #f8fafc; padding: 8px 12px; border-radius: 6px; color: #4a5568; font-size: 0.9rem;">
                    ✔ Passed all operational constraints — Executed as proposed
                </div>
            </div>

            <div style="margin-top: 12px; padding-top: 10px; border-top: 1px dashed #cbd5e0; font-size: 0.85rem; color: #4a5568;">
                <strong>🔄 Learning Loop:</strong> Status — <span style="color: #3182ce; font-weight: bold;">${item.feedback_queue || 'QUEUED_FOR_REVIEW'}</span>
            </div>
        `;
    }
    
    document.getElementById('modal-summary').innerHTML = summaryHTML;
    
    // 2. Model Explanation (SHAP)
    let reasonHTML = `<div style="font-family: monospace; background: #f8fafc; padding: 15px; border-radius: 6px; border: 1px solid #e2e8f0;">`;
    const proposedAction = item.original_action || item.recommended_action;
    reasonHTML += `<strong style="font-family: sans-serif;">Why did the AI propose ${toTitleCase(proposedAction)}?</strong><br><br>`;

    if (item.shap_drivers) {
        if (item.shap_drivers.positive && item.shap_drivers.positive.length > 0) {
            reasonHTML += `<em style="color: #4a5568;">Positive model drivers</em><br>`;
            item.shap_drivers.positive.forEach(d => {
                reasonHTML += `<span style="color: #38a169;">↑ ${d.feature.padEnd(25, ' ')} +${d.impact.toFixed(2)}</span><br>`;
            });
        }
        
        if (item.shap_drivers.negative && item.shap_drivers.negative.length > 0) {
            reasonHTML += `<br><em style="color: #4a5568;">Negative model drivers</em><br>`;
            item.shap_drivers.negative.forEach(d => {
                reasonHTML += `<span style="color: #e53e3e;">↓ ${d.feature.padEnd(25, ' ')} −${Math.abs(d.impact).toFixed(2)}</span><br>`;
            });
        }
    } else {
        reasonHTML += `<span>No feature attribution data recorded.</span>`;
    }
    reasonHTML += `</div>`;
    document.getElementById('modal-reason').innerHTML = reasonHTML;
    
    // 3. Guardrails Status
    const gl = document.getElementById('modal-guardrails');
    if (item.action_overridden) {
        gl.innerHTML = `<li style="color: #e53e3e;">⚠ Guardrail override active (see flow above)</li>`;
    } else {
        gl.innerHTML = `<li>✔ Passed all operational constraints</li>`;
    }
    
    // 4. Counterfactual action evaluation (Labeled with 🤖 Proposed and 🔴 Enforced)
    const alt = document.getElementById('modal-alternatives');
    let altsHTML = "";
    
    // Build full list including the final action and alternatives to sort by expected revenue cleanly
    let allEvaluatedActions = [];
    
    // Add recommended (final) action
    allEvaluatedActions.push({
        action: item.recommended_action,
        probability: item.predicted_recovery_probability || item.predicted_probability || 0,
        expected_revenue: item.expected_revenue || 0,
        isFinal: true,
        isProposedAI: (item.recommended_action === item.original_action)
    });
    
    // Add alternatives
    if (item.alternative_actions) {
        item.alternative_actions.forEach(a => {
            allEvaluatedActions.push({
                action: a.action,
                probability: a.probability,
                expected_revenue: a.expected_revenue,
                isFinal: false,
                isProposedAI: (a.action === item.original_action)
            });
        });
    }
    
    // Sort descending by expected revenue
    allEvaluatedActions.sort((a, b) => b.expected_revenue - a.expected_revenue);

    allEvaluatedActions.forEach(rowItem => {
        let labelTag = "";
        let rowStyle = "";
        
        if (rowItem.isFinal) {
            rowStyle = "background: #f0fff4; font-weight: bold;";
            labelTag = " 🔴 Enforced";
        } else if (rowItem.isProposedAI) {
            labelTag = " 🤖 Proposed";
        }

        altsHTML += `
            <tr style="${rowStyle}">
                <td>${toTitleCase(rowItem.action)}${labelTag}</td>
                <td>${safeProb(rowItem.probability)}%</td>
                <td>₹${rowItem.expected_revenue.toLocaleString('en-IN', {minimumFractionDigits: 2})}</td>
            </tr>
        `;
    });
    
    alt.innerHTML = altsHTML;

    // Display the modal
    document.getElementById('decision-modal').style.display = "block";
}

function closeModal() {
    document.getElementById('decision-modal').style.display = "none";
}

// Close modal if clicked outside
window.onclick = function(event) {
    const modal = document.getElementById('decision-modal');
    if (event.target == modal) {
        modal.style.display = "none";
    }
}

fetchDecisions();
setInterval(fetchDecisions, 2500);