#pragma once

#include <tinympc/types.hpp>

tinytype rho_value = 5.0;

tinytype Adyn_data[NSTATES*NSTATES] = {
  1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0001570,	-0.0000000,	0.0040000,	0.0000000,	0.0000000,	0.0000000,	0.0000001,	-0.0000000,	
  0.0000000,	1.0000000,	0.0000000,	-0.0001570,	0.0000000,	-0.0000000,	0.0000000,	0.0040000,	0.0000000,	-0.0000001,	0.0000000,	-0.0000000,	
  0.0000000,	0.0000000,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0040000,	0.0000000,	0.0000000,	0.0000000,	
  0.0000000,	0.0000000,	0.0000000,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0020000,	-0.0000000,	0.0000000,	
  0.0000000,	0.0000000,	0.0000000,	-0.0000000,	1.0000000,	-0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0020000,	-0.0000000,	
  0.0000000,	0.0000000,	0.0000000,	-0.0000000,	0.0000000,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-0.0000000,	0.0000000,	0.0020000,	
  0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0784800,	-0.0000000,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000785,	-0.0000000,	
  0.0000000,	0.0000000,	0.0000000,	-0.0784800,	0.0000000,	-0.0000000,	0.0000000,	1.0000000,	0.0000000,	-0.0000785,	0.0000000,	-0.0000000,	
  0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	1.0000000,	-0.0000000,	0.0000000,	
  0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	1.0000000,	-0.0000000,	
  0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	-0.0000000,	0.0000000,	1.0000000	
};

tinytype Bdyn_data[NSTATES*NINPUTS] = {
  -0.0000000,	0.0000000,	0.0000000,	-0.0000000,	
  0.0000000,	0.0000000,	-0.0000000,	-0.0000000,	
  0.0000336,	0.0000336,	0.0000336,	0.0000336,	
  -0.0011014,	-0.0012129,	0.0011027,	0.0012117,	
  -0.0011068,	0.0012171,	0.0011103,	-0.0012206,	
  0.0000790,	-0.0000289,	-0.0001114,	0.0000613,	
  -0.0000290,	0.0000318,	0.0000290,	-0.0000319,	
  0.0000288,	0.0000317,	-0.0000288,	-0.0000317,	
  0.0168171,	0.0168171,	0.0168171,	0.0168171,	
  -1.1014184,	-1.2129361,	1.1026505,	1.2117040,	
  -1.1068281,	1.2171137,	1.1102780,	-1.2205636,	
  0.0789908,	-0.0288946,	-0.1113750,	0.0612788	
};

tinytype Kinf_data[NINPUTS*NSTATES] = {
  -0.0488908,	0.0489869,	0.0510852,	-0.2303115,	-0.2341621,	-0.1007044,	-0.0366831,	0.0366776,	0.0793666,	-0.0211353,	-0.0240878,	-0.0808721,	
  0.0487535,	0.0489476,	0.0510852,	-0.2205505,	0.2295610,	0.1004956,	0.0363103,	0.0362812,	0.0793666,	-0.0165504,	0.0234384,	0.0806128,	
  0.0491580,	-0.0490932,	0.0510852,	0.2251870,	0.2213453,	-0.0999705,	0.0366398,	-0.0366602,	0.0793666,	0.0172022,	0.0142541,	-0.0799635,	
  -0.0490208,	-0.0488413,	0.0510852,	0.2256750,	-0.2167441,	0.1001793,	-0.0362670,	-0.0362986,	0.0793666,	0.0204835,	-0.0136047,	0.0802228	
};

tinytype Quu_inv_data[NINPUTS*NINPUTS] = {
  0.0001000,	0.0000000,	0.0000000,	0.0000000,	
  0.0000000,	0.0001000,	0.0000000,	0.0000000,	
  0.0000000,	0.0000000,	0.0001000,	0.0000000,	
  0.0000000,	0.0000000,	0.0000000,	0.0001000	
};

tinytype AmBKt_data[NSTATES*NSTATES] = {
  1.0000000,	-0.0000000,	-0.0000000,	0.0000105,	-0.0002279,	0.0000137,	-0.0000060,	-0.0000003,	-0.0000000,	0.0104803,	-0.2278644,	0.0137495,	
  -0.0000000,	1.0000000,	-0.0000000,	0.0002266,	-0.0000105,	-0.0000049,	-0.0000003,	-0.0000059,	-0.0000000,	0.2266394,	-0.0104615,	-0.0049300,	
  -0.0000000,	-0.0000000,	0.9999931,	0.0000000,	-0.0000000,	0.0000000,	-0.0000000,	-0.0000000,	-0.0034364,	0.0000000,	-0.0000000,	0.0000000,	
  0.0000000,	-0.0001569,	0.0000000,	0.9989571,	0.0000390,	0.0000231,	0.0000010,	-0.0784527,	0.0000000,	-1.0429368,	0.0389503,	0.0230709,	
  0.0001569,	-0.0000000,	-0.0000000,	0.0000391,	0.9989511,	0.0000631,	0.0784526,	-0.0000010,	-0.0000000,	0.0390957,	-1.0488838,	0.0630639,	
  -0.0000000,	0.0000000,	0.0000000,	-0.0000002,	-0.0000005,	0.9999936,	-0.0000000,	0.0000000,	0.0000000,	-0.0001781,	-0.0005068,	-0.0064146,	
  0.0040000,	-0.0000000,	-0.0000000,	0.0000072,	-0.0001697,	0.0000102,	0.9999956,	-0.0000002,	-0.0000000,	0.0071825,	-0.1697422,	0.0102500,	
  -0.0000000,	0.0040000,	-0.0000000,	0.0001688,	-0.0000072,	-0.0000037,	-0.0000002,	0.9999956,	-0.0000000,	0.1688108,	-0.0071643,	-0.0037076,	
  -0.0000000,	-0.0000000,	0.0039893,	0.0000000,	-0.0000000,	0.0000000,	-0.0000000,	-0.0000000,	0.9946611,	0.0000000,	-0.0000000,	0.0000000,	
  0.0000000,	-0.0000001,	0.0000000,	0.0019129,	0.0000027,	0.0000019,	0.0000001,	-0.0000762,	0.0000000,	0.9128586,	0.0026528,	0.0018520,	
  0.0000001,	-0.0000000,	0.0000000,	0.0000027,	0.0019124,	0.0000050,	0.0000762,	-0.0000001,	0.0000000,	0.0026660,	0.9123803,	0.0050012,	
  -0.0000000,	0.0000000,	0.0000000,	-0.0000003,	-0.0000009,	0.0019949,	-0.0000000,	0.0000000,	0.0000000,	-0.0003303,	-0.0009278,	0.9948955	
};

tinytype coeff_d2p_data[NSTATES*NINPUTS] = {
  -489.1523203,	487.7790877,	491.8262134,	-490.4529797,	
  490.1143119,	489.7209258,	-491.1779152,	-488.6573225,	
  511.1075132,	511.1075297,	511.1075287,	511.1075143,	
  -2304.2660646,	-2206.6077802,	2252.9958380,	2257.8780066,	
  -2342.7919797,	2296.7579293,	2214.5592570,	-2168.5252047,	
  -1007.5478488,	1005.4585331,	-1000.2049257,	1002.2942410,	
  -367.0142696,	363.2841303,	366.5815107,	-362.8513710,	
  366.9595161,	362.9936085,	-366.7854221,	-363.1677025,	
  794.0631100,	794.0631434,	794.0631309,	794.0631237,	
  -211.4583209,	-165.5871552,	172.1082045,	204.9372717,	
  -240.9988903,	234.5011629,	142.6124360,	-136.1147087,	
  -809.1253217,	806.5310209,	-800.0344495,	802.6287494	
};

tinytype Q_data[NSTATES]= {100.0000000,	100.0000000,	100.0000000,	4.0000000,	4.0000000,	400.0000000,	4.0000000,	4.0000000,	4.0000000,	2.0408163,	2.0408163,	4.0000000};

tinytype Qf_data[NSTATES]= {100.0000000,	100.0000000,	100.0000000,	4.0000000,	4.0000000,	400.0000000,	4.0000000,	4.0000000,	4.0000000,	2.0408163,	2.0408163,	4.0000000};

tinytype R_data[NINPUTS]= {10000.0000000,	10000.0000000,	10000.0000000,	10000.0000000};
