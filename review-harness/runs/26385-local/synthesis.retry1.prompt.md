# Role

You assemble the final deliverables by SELECTING findings verbatim. You are forbidden to
rewrite, soften, or drop finding content. Two deterministic auditors will diff your output
against every input finding and force restoration of anything missing.

# Inputs (absolute paths) and input-id namespaces

- /Users/dabs/Development/scikit-learn/review-harness/runs/26385-local/depth.md    — ids D:F<n> (e.g. block "### F3" in depth.md is D:F3)
- /Users/dabs/Development/scikit-learn/review-harness/runs/26385-local/closure.md  — ids C:F<n>
- /Users/dabs/Development/scikit-learn/review-harness/runs/26385-local/inverse.md  — ids I:F<n>

A file containing only NO FINDINGS contributes zero ids.

# Task — write THREE files with the Write tool (absolute paths)

1. /Users/dabs/Development/scikit-learn/review-harness/runs/26385-local/report.md — human-first report:
   - header: PR 26385, range 86541f2b3bc8a96264e265cb810cf79858544340..04c8b6954e3e4b8f8af086cea9d386f954b76bfe, counts by severity
   - findings grouped high → medium → low; each rendered as a severity-prefixed one-line
     title, then 2–4 plain sentences a maintainer can act on, then the verbatim block(s)
     inside a collapsed section:
     <details><summary>verbatim finding</summary>

     ```
     <original block(s), byte-identical>
     ```
     </details>
2. /Users/dabs/Development/scikit-learn/review-harness/runs/26385-local/findings.json — JSON array; one object per delivered unit:
   {"id": "U<k>", "severity": "...", "title": "...", "file": "...", "line": <int>,
    "scenario": "...", "contract": "...", "instances": [...], "body": "<verbatim block(s)>"}
   (file/line = the primary evidence anchor; instances = the block's instances list, or
   [file:line] for single-instance.)
3. /Users/dabs/Development/scikit-learn/review-harness/runs/26385-local/synthesis_ledger.json — JSON object mapping EVERY input finding id to
   {"disposition": "kept-as" | "merged-into" | "duplicate-of", "unit": "U<k>"}.
   kept-as: the unit is this finding verbatim. duplicate-of: byte-near-identical defect
   already delivered as unit U<k>. merged-into: grouped with siblings into unit U<k>.

# Rules

- Every input id appears in the ledger; every ledger unit exists in findings.json.
- Dedup only truly identical defects (same root cause, same site). When merging, keep the
  most specific member's wording verbatim and CARRY EVERY MEMBER'S evidence anchors into
  the unit body — the auditor rejects units that lost a member's anchors.
- No merge group larger than 6 members. At least 60% of input findings must survive as
  delivered units.
- Order findings.json by severity (high first), then by file path.

When the three files are written, reply with exactly: SYNTHESIS DONE


# RETRY NOTICE — the deterministic auditors rejected your output
PRESERVATION: unit U1 lost all evidence anchors of member D:F77
PRESERVATION: unit U100 lost all evidence anchors of member D:F143
PRESERVATION: unit U101 lost all evidence anchors of member D:F144
PRESERVATION: unit U102 lost all evidence anchors of member D:F145
PRESERVATION: unit U103 lost all evidence anchors of member D:F147
PRESERVATION: unit U104 lost all evidence anchors of member D:F148
PRESERVATION: unit U105 lost all evidence anchors of member D:F150
PRESERVATION: unit U106 lost all evidence anchors of member D:F151
PRESERVATION: unit U107 lost all evidence anchors of member D:F152
PRESERVATION: unit U108 lost all evidence anchors of member D:F153
PRESERVATION: unit U109 lost all evidence anchors of member D:F157
PRESERVATION: unit U11 lost all evidence anchors of member D:F42
PRESERVATION: unit U11 lost all evidence anchors of member D:F43
PRESERVATION: unit U11 lost all evidence anchors of member D:F53
PRESERVATION: unit U110 lost all evidence anchors of member D:F158
PRESERVATION: unit U111 lost all evidence anchors of member D:F159
PRESERVATION: unit U112 lost all evidence anchors of member D:F160
PRESERVATION: unit U113 lost all evidence anchors of member D:F161
PRESERVATION: unit U114 lost all evidence anchors of member D:F162
PRESERVATION: unit U115 lost all evidence anchors of member D:F163
PRESERVATION: unit U116 lost all evidence anchors of member D:F164
PRESERVATION: unit U117 lost all evidence anchors of member D:F165
PRESERVATION: unit U118 lost all evidence anchors of member D:F168
PRESERVATION: unit U119 lost all evidence anchors of member D:F169
PRESERVATION: unit U120 lost all evidence anchors of member D:F171
PRESERVATION: unit U121 lost all evidence anchors of member D:F173
PRESERVATION: unit U122 lost all evidence anchors of member D:F174
PRESERVATION: unit U123 lost all evidence anchors of member D:F175
PRESERVATION: unit U124 lost all evidence anchors of member D:F176
PRESERVATION: unit U125 lost all evidence anchors of member D:F177
PRESERVATION: unit U126 lost all evidence anchors of member D:F178
PRESERVATION: unit U127 lost all evidence anchors of member D:F179
PRESERVATION: unit U128 lost all evidence anchors of member D:F180
PRESERVATION: unit U129 lost all evidence anchors of member D:F182
PRESERVATION: unit U130 lost all evidence anchors of member D:F183
PRESERVATION: unit U131 lost all evidence anchors of member D:F184
PRESERVATION: unit U132 lost all evidence anchors of member D:F185
PRESERVATION: unit U133 lost all evidence anchors of member D:F187
PRESERVATION: unit U134 lost all evidence anchors of member D:F188
PRESERVATION: unit U135 lost all evidence anchors of member D:F189
PRESERVATION: unit U136 lost all evidence anchors of member D:F190
PRESERVATION: unit U137 lost all evidence anchors of member D:F191
PRESERVATION: unit U138 lost all evidence anchors of member D:F192
PRESERVATION: unit U139 lost all evidence anchors of member D:F193
PRESERVATION: unit U140 lost all evidence anchors of member D:F194
PRESERVATION: unit U141 lost all evidence anchors of member D:F195
PRESERVATION: unit U142 lost all evidence anchors of member D:F196
PRESERVATION: unit U143 lost all evidence anchors of member D:F197
PRESERVATION: unit U144 lost all evidence anchors of member D:F198
PRESERVATION: unit U145 lost all evidence anchors of member D:F199
PRESERVATION: unit U146 lost all evidence anchors of member D:F200
PRESERVATION: unit U147 lost all evidence anchors of member D:F201
PRESERVATION: unit U148 lost all evidence anchors of member D:F202
PRESERVATION: unit U149 lost all evidence anchors of member D:F203
PRESERVATION: unit U150 lost all evidence anchors of member D:F204
PRESERVATION: unit U151 lost all evidence anchors of member D:F205
PRESERVATION: unit U152 lost all evidence anchors of member D:F206
PRESERVATION: unit U153 lost all evidence anchors of member D:F207
PRESERVATION: unit U154 lost all evidence anchors of member D:F208
PRESERVATION: unit U155 lost all evidence anchors of member D:F210
PRESERVATION: unit U156 lost all evidence anchors of member D:F212
PRESERVATION: unit U157 lost all evidence anchors of member D:F214
PRESERVATION: unit U158 lost all evidence anchors of member D:F215
PRESERVATION: unit U159 lost all evidence anchors of member D:F216
PRESERVATION: unit U160 lost all evidence anchors of member D:F217
PRESERVATION: unit U161 lost all evidence anchors of member D:F218
PRESERVATION: unit U162 lost all evidence anchors of member D:F219
PRESERVATION: unit U163 lost all evidence anchors of member D:F220
PRESERVATION: unit U164 lost all evidence anchors of member D:F222
PRESERVATION: unit U165 lost all evidence anchors of member D:F223
PRESERVATION: unit U166 lost all evidence anchors of member D:F224
PRESERVATION: unit U167 lost all evidence anchors of member D:F227
PRESERVATION: unit U168 lost all evidence anchors of member D:F228
PRESERVATION: unit U169 lost all evidence anchors of member D:F229
PRESERVATION: unit U170 lost all evidence anchors of member D:F230
PRESERVATION: unit U171 lost all evidence anchors of member D:F231
PRESERVATION: unit U172 lost all evidence anchors of member D:F232
PRESERVATION: unit U173 lost all evidence anchors of member D:F233
PRESERVATION: unit U174 lost all evidence anchors of member D:F238
PRESERVATION: unit U175 lost all evidence anchors of member D:F239
PRESERVATION: unit U176 lost all evidence anchors of member D:F241
PRESERVATION: unit U177 lost all evidence anchors of member D:F242
PRESERVATION: unit U2 merge group size 7 > 6
PRESERVATION: unit U2 lost all evidence anchors of member D:F167
PRESERVATION: unit U21 lost all evidence anchors of member D:F18
PRESERVATION: unit U21 lost all evidence anchors of member D:F22
PRESERVATION: unit U21 lost all evidence anchors of member D:F40
PRESERVATION: unit U22 lost all evidence anchors of member D:F138
PRESERVATION: unit U22 lost all evidence anchors of member D:F55
PRESERVATION: unit U23 lost all evidence anchors of member D:F125
PRESERVATION: unit U23 lost all evidence anchors of member D:F155
PRESERVATION: unit U23 lost all evidence anchors of member D:F225
PRESERVATION: unit U23 lost all evidence anchors of member D:F236
PRESERVATION: unit U23 lost all evidence anchors of member D:F70
PRESERVATION: unit U24 lost all evidence anchors of member D:F112
PRESERVATION: unit U24 lost all evidence anchors of member D:F128
PRESERVATION: unit U24 lost all evidence anchors of member D:F154
PRESERVATION: unit U24 lost all evidence anchors of member D:F97
PRESERVATION: unit U25 lost all evidence anchors of member D:F126
PRESERVATION: unit U25 lost all evidence anchors of member D:F221
PRESERVATION: unit U25 lost all evidence anchors of member D:F235
PRESERVATION: unit U25 lost all evidence anchors of member D:F44
PRESERVATION: unit U25 lost all evidence anchors of member D:F76
PRESERVATION: unit U26 lost all evidence anchors of member D:F114
PRESERVATION: unit U26 lost all evidence anchors of member D:F156
PRESERVATION: unit U26 lost all evidence anchors of member D:F181
PRESERVATION: unit U26 lost all evidence anchors of member D:F237
PRESERVATION: unit U27 lost all evidence anchors of member D:F116
PRESERVATION: unit U27 lost all evidence anchors of member D:F134
PRESERVATION: unit U27 lost all evidence anchors of member D:F240
PRESERVATION: unit U28 lost all evidence anchors of member D:F105
PRESERVATION: unit U28 lost all evidence anchors of member D:F123
PRESERVATION: unit U28 lost all evidence anchors of member D:F226
PRESERVATION: unit U3 lost all evidence anchors of member D:F72
PRESERVATION: unit U34 lost all evidence anchors of member D:F20
PRESERVATION: unit U35 lost all evidence anchors of member D:F24
PRESERVATION: unit U36 lost all evidence anchors of member D:F25
PRESERVATION: unit U37 lost all evidence anchors of member D:F26
PRESERVATION: unit U38 lost all evidence anchors of member D:F27
PRESERVATION: unit U39 lost all evidence anchors of member D:F29
PRESERVATION: unit U4 lost all evidence anchors of member D:F166
PRESERVATION: unit U4 lost all evidence anchors of member D:F75
PRESERVATION: unit U40 lost all evidence anchors of member D:F30
PRESERVATION: unit U41 lost all evidence anchors of member D:F31
PRESERVATION: unit U42 lost all evidence anchors of member D:F32
PRESERVATION: unit U43 lost all evidence anchors of member D:F33
PRESERVATION: unit U44 lost all evidence anchors of member D:F35
PRESERVATION: unit U45 lost all evidence anchors of member D:F38
PRESERVATION: unit U46 lost all evidence anchors of member D:F41
PRESERVATION: unit U47 lost all evidence anchors of member D:F46
PRESERVATION: unit U48 lost all evidence anchors of member D:F47
PRESERVATION: unit U49 lost all evidence anchors of member D:F48
PRESERVATION: unit U50 lost all evidence anchors of member D:F49
PRESERVATION: unit U51 lost all evidence anchors of member D:F51
PRESERVATION: unit U52 lost all evidence anchors of member D:F52
PRESERVATION: unit U53 lost all evidence anchors of member D:F56
PRESERVATION: unit U54 lost all evidence anchors of member D:F58
PRESERVATION: unit U55 lost all evidence anchors of member D:F59
PRESERVATION: unit U56 lost all evidence anchors of member D:F60
PRESERVATION: unit U57 lost all evidence anchors of member D:F61
PRESERVATION: unit U58 lost all evidence anchors of member D:F62
PRESERVATION: unit U59 lost all evidence anchors of member D:F64
PRESERVATION: unit U6 lost all evidence anchors of member D:F149
PRESERVATION: unit U60 lost all evidence anchors of member D:F65
PRESERVATION: unit U61 lost all evidence anchors of member D:F66
PRESERVATION: unit U62 lost all evidence anchors of member D:F67
PRESERVATION: unit U63 lost all evidence anchors of member D:F68
PRESERVATION: unit U64 lost all evidence anchors of member D:F69
PRESERVATION: unit U65 lost all evidence anchors of member D:F71
PRESERVATION: unit U66 lost all evidence anchors of member D:F73
PRESERVATION: unit U67 lost all evidence anchors of member D:F74
PRESERVATION: unit U68 lost all evidence anchors of member D:F78
PRESERVATION: unit U69 lost all evidence anchors of member D:F82
PRESERVATION: unit U70 lost all evidence anchors of member D:F83
PRESERVATION: unit U71 lost all evidence anchors of member D:F84
PRESERVATION: unit U72 lost all evidence anchors of member D:F85
PRESERVATION: unit U73 lost all evidence anchors of member D:F86
PRESERVATION: unit U74 lost all evidence anchors of member D:F88
PRESERVATION: unit U75 lost all evidence anchors of member D:F89
PRESERVATION: unit U76 lost all evidence anchors of member D:F91
PRESERVATION: unit U77 lost all evidence anchors of member D:F92
PRESERVATION: unit U78 lost all evidence anchors of member D:F93
PRESERVATION: unit U79 lost all evidence anchors of member D:F94
PRESERVATION: unit U8 merge group size 7 > 6
PRESERVATION: unit U8 lost all evidence anchors of member D:F54
PRESERVATION: unit U80 lost all evidence anchors of member D:F95
PRESERVATION: unit U81 lost all evidence anchors of member D:F96
PRESERVATION: unit U82 lost all evidence anchors of member D:F98
PRESERVATION: unit U83 lost all evidence anchors of member D:F100
PRESERVATION: unit U84 lost all evidence anchors of member D:F101
PRESERVATION: unit U85 lost all evidence anchors of member D:F107
PRESERVATION: unit U86 lost all evidence anchors of member D:F108
PRESERVATION: unit U87 lost all evidence anchors of member D:F109
PRESERVATION: unit U88 lost all evidence anchors of member D:F110
PRESERVATION: unit U89 lost all evidence anchors of member D:F111
PRESERVATION: unit U9 merge group size 7 > 6
PRESERVATION: unit U9 lost all evidence anchors of member D:F28
PRESERVATION: unit U90 lost all evidence anchors of member D:F113
PRESERVATION: unit U91 lost all evidence anchors of member D:F115
PRESERVATION: unit U92 lost all evidence anchors of member D:F122
PRESERVATION: unit U93 lost all evidence anchors of member D:F124
PRESERVATION: unit U94 lost all evidence anchors of member D:F127
PRESERVATION: unit U95 lost all evidence anchors of member D:F129
PRESERVATION: unit U96 lost all evidence anchors of member D:F135
PRESERVATION: unit U97 lost all evidence anchors of member D:F136
PRESERVATION: unit U98 lost all evidence anchors of member D:F139
PRESERVATION: unit U99 lost all evidence anchors of member D:F141
missing_ids: D:F100, D:F101, D:F105, D:F107, D:F108, D:F109, D:F110, D:F111, D:F112, D:F113, D:F114, D:F115, D:F116, D:F122, D:F123, D:F124, D:F125, D:F126, D:F127, D:F128, D:F129, D:F134, D:F135, D:F136, D:F138, D:F139, D:F141, D:F143, D:F144, D:F145, D:F147, D:F148, D:F149, D:F150, D:F151, D:F152, D:F153, D:F154, D:F155, D:F156, D:F157, D:F158, D:F159, D:F160, D:F161, D:F162, D:F163, D:F164, D:F165, D:F166, D:F167, D:F168, D:F169, D:F171, D:F173, D:F174, D:F175, D:F176, D:F177, D:F178, D:F179, D:F18, D:F180, D:F181, D:F182, D:F183, D:F184, D:F185, D:F187, D:F188, D:F189, D:F190, D:F191, D:F192, D:F193, D:F194, D:F195, D:F196, D:F197, D:F198, D:F199, D:F20, D:F200, D:F201, D:F202, D:F203, D:F204, D:F205, D:F206, D:F207, D:F208, D:F210, D:F212, D:F214, D:F215, D:F216, D:F217, D:F218, D:F219, D:F22, D:F220, D:F221, D:F222, D:F223, D:F224, D:F225, D:F226, D:F227, D:F228, D:F229, D:F230, D:F231, D:F232, D:F233, D:F235, D:F236, D:F237, D:F238, D:F239, D:F24, D:F240, D:F241, D:F242, D:F25, D:F26, D:F27, D:F28, D:F29, D:F30, D:F31, D:F32, D:F33, D:F35, D:F38, D:F40, D:F41, D:F42, D:F43, D:F44, D:F46, D:F47, D:F48, D:F49, D:F51, D:F52, D:F53, D:F54, D:F55, D:F56, D:F58, D:F59, D:F60, D:F61, D:F62, D:F64, D:F65, D:F66, D:F67, D:F68, D:F69, D:F70, D:F71, D:F72, D:F73, D:F74, D:F75, D:F76, D:F77, D:F78, D:F82, D:F83, D:F84, D:F85, D:F86, D:F88, D:F89, D:F91, D:F92, D:F93, D:F94, D:F95, D:F96, D:F97, D:F98

Restore every missing/mishandled input finding and rewrite ALL THREE files.
