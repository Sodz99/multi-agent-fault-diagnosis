# Incident Postmortem: East Cluster Degradation

## Incident Summary
- **Incident ID**: INC-2024-03-456
- **Date**: March 15, 2024
- **Duration**: 2h 34m (14:22 - 16:56 UTC)
- **Severity**: P1 - Critical Service Impact
- **Affected Services**: Voice calls, Data sessions in East Cluster
- **Customer Impact**: 12,000 subscribers affected

## Timeline

### 14:22 UTC - Initial Detection
- Automated monitoring detected packet loss spike (8.7%) on RAN-SITE-12A
- NOC alerted via PagerDuty escalation
- Call drop rate exceeded 15% threshold

### 14:25 UTC - Escalation to L2
- NOC engineer confirmed transport congestion on TL-E3-001
- Link utilization peaked at 99.8%
- MME-CLSTR-3 showing S1-AP timeout errors

### 14:31 UTC - Investigation Findings
- Root cause identified: Scheduled fiber maintenance on primary route
- Traffic automatically failed over to backup microwave link TL-E3-002
- Backup link insufficient capacity (100Mbps vs 1Gbps primary)
- No advance notification received from transport team

### 14:45 UTC - Mitigation Attempts
1. **Load Balancing**: Redirected 40% traffic to neighbor sites 11A, 13A
2. **Admission Control**: Reduced new call acceptance to 60%
3. **QoS Adjustment**: Throttled non-voice traffic by 50%
4. **Emergency Capacity**: Activated temporary microwave link TL-E3-003

### 15:20 UTC - Partial Recovery
- Packet loss reduced to 2.1%
- Call drop rate decreased to 4.8%
- Still above acceptable thresholds but service partially restored

### 16:56 UTC - Full Resolution
- Primary fiber link TL-E3-001 restored
- All KPIs returned to normal baseline
- Load balancing configurations reverted to standard

## Root Cause Analysis

### Primary Cause
Unscheduled fiber maintenance by transport vendor without proper change management notification. The maintenance window was moved from 02:00 (low traffic) to 14:00 (peak busy hour) without updating stakeholders.

### Contributing Factors
1. **Insufficient Backup Capacity**: Microwave backup only 10% of primary capacity
2. **Monitoring Gap**: No proactive alerting on transport link utilization trends
3. **Change Management**: Communication breakdown between transport and network teams
4. **Documentation**: Outdated network diagram showed incorrect backup link capacity

## Corrective Actions

### Immediate (Completed)
- [ ] Restore primary transport link ✅
- [ ] Validate backup link capacity specifications ✅
- [ ] Update network topology documentation ✅
- [ ] Verify all monitoring thresholds ✅

### Short-term (Within 30 days)
- [ ] Implement transport link utilization trending alerts
- [ ] Upgrade backup microwave links to 500Mbps minimum
- [ ] Establish weekly coordination calls with transport team
- [ ] Deploy additional monitoring probes on critical links

### Long-term (Within 90 days)
- [ ] Design redundant fiber routes for all critical sites
- [ ] Implement automated traffic engineering for congestion
- [ ] Develop predictive analytics for capacity planning
- [ ] Create runbook automation for common failure scenarios

## Lessons Learned

### What Went Well
- Automated detection systems triggered within 3 minutes
- NOC response time met SLA targets
- Load balancing partially mitigated impact
- Customer communication was timely and accurate

### What Could Be Improved
- Transport vendor change management integration
- Backup capacity planning assumptions
- Real-time topology awareness
- Automated remediation capabilities

## Prevention Measures

### Monitoring Enhancements
- Add transport link capacity utilization alerts at 75%, 85%, 95%
- Implement predictive alerting based on traffic growth trends
- Deploy synthetic transaction monitoring for end-to-end validation
- Create correlation rules between transport and RAN KPIs

### Process Improvements
- Mandatory change approval for all transport maintenance
- Daily capacity reports for critical network segments
- Monthly disaster recovery testing of backup links
- Quarterly network topology validation and documentation updates

## Financial Impact
- **Revenue Loss**: $47,000 (voice/data service credits)
- **Operational Cost**: $12,000 (emergency technician dispatch)
- **Total Impact**: $59,000

---
*Postmortem compiled by: Senior Network Engineer*
*Reviewed by: Network Operations Manager*
*Approved by: VP Network Engineering*