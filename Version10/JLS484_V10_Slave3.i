SO 3887 JLS484 Replacement, Inventor Version 10
c -------------------------------------------------------------------
c
c   Replacement for the JLS484 irradiator at John Hopkins
c   Similar Geometry, 26000 Ci of Co60 2 source rods 4 pellets
c   
c   Reminder:
c   Currently modeling the main shield and specifically the beam chamber
c   have not added the angled side to the beam chanber
c
c -------------------------------------------------------------------
c
c                       Cell Cards
c
c -------------------------------------------------------------------
c       Source Rods
c -------------------------------------------------------------------
998 99 -0.001205 -999
c
1 1 -8.860000 -100:-101:-108:-109 $ Sources
2 100 -8.000000 (-102 100):(-103 101):(-110 108):(-111 109):-104:-105:-112:-113 $ Cladding
3 100 -8.000000 (-107 106):(-115 114):-128:-129:-130:-132:-133:-134: $ SS Source Rod
        (-124 119):(-125 116 117 118 119):(-126 123):(-127 120 121 122 123)
4 300 -8.400000 (-131 130):(-135 134) $ Bronze Bearing
5 200 -19.300000 -136:-137:-138:-139:-140:-141:-142:-143:-144:-145 $ Lower Tungsten Rod
6 300 -8.400000 (-146 140):(-147 145) $Bronze Bearing
7 200 -19.300000 -148:-149:-150:-151:-152:-153
c -------------------------------------------------------------------
c       Primary Shield
c -------------------------------------------------------------------
10 100 -8.000000 (-201 200):(-203 202) $ Source Rod Tubes
11 99 -0.001205 -204 -206 -207 201 203 $Inside Beam Chamber
12 100 -8.000000 (-205 -208 -209) (204:206:207) 201 203 $SS Chamber
13 400 -11.350000 (-210 -211 -212 -213 -214 -215 -216 201 203 ):(-225 -213 -214 -215 -216 201 203) #11 #12 $ Pb Main
14 100 -8.000000 (-217 -218 -219 -220 -221 -222 -223 201 203 #13):(-226 -218 -219 -220 -221 -222 -223 201 203 #11 #12 #13)$ SS Main
15 100 -8.000000 (-224 -218 -219)
c -------------------------------------------------------------------
c       Secondary Shield
c -------------------------------------------------------------------
20 400 -11.350000 (-250 -251 -252 -253 -254 #23 #22 270 #25 #26 #27):(-255 #23 #22 272 #25 #26 #27 291 #28 #29) $ Lead
21 100 -8.000000 (-256 -257 -258 -259 -260 #20 #22 270 #25 #26 #27):(-261 #20 #22 272 #25 #25 #27 291 #28 #29) $ Steel Liner
22 99 -0.001205 -265 #25 #26 #27 $ Air in Chamber
23 100 -8.000000 -266 265 270 272 #25 #26 #27 #28 #29 $ Stell Chamber Liner
24 100 -8.000000 (-270 271 -260):(-272 273) $ Cable Ports
25 99 -0.001205 (-276 278):(-280 282):(-284)$ Door Space
26 100 -8.000000 (-275 277 #25):(-279 281 #25):(-283 -259 #25) $ side Door Liner
27 100 -8.000000 (-290 291 265) #28$ Front Door
28 99 -0.001205 (-293 295):(-292 -294)
29 100 -8.000000 (-292 294 290) #28
c -------------------------------------------------------------------
c       Pedestal
c -------------------------------------------------------------------
30 100 -8.000000 -302:(-301 300 258)
c -------------------------------------------------------------------
c       Side Door
c -------------------------------------------------------------------
31 400 -11.350000 (-310 312):(-314 316):(-318 -254)
32 100 -8.000000 (-311 313 #31):(-315 317 #31):(-319 -259 #31)
c -------------------------------------------------------------------
c       Front Door
c -------------------------------------------------------------------
33 400 -11.350000 (-320 322):(-324 326):-328
34 100 -8.000000 (-321 323 #33):(-325 327 #33):(-329 #33)
c -------------------------------------------------------------------
c       Cable Baffle
c -------------------------------------------------------------------
35 400 -11.350000 -340:-341:-342:-343
36 100 -8.000000 (-344:-345:-346:-347) #35
c -------------------------------------------------------------------
c       Port Plugs
c -------------------------------------------------------------------
37 200 -19.300000 (-350:-351) $ Front Top port
38 200 -19.300000 (-352 -260):(-353)
c
999 0 999

c -------------------------------------------------------------------
c
c                       Surface Cards
c
c -------------------------------------------------------------------
c       Source Rods
c -------------------------------------------------------------------
100 RCC 0 4.45 2.959 0 0 2.972 0.9905   $ +Y/+Z Source
101 RCC 0 4.45 -2.959 0 0 -2.972 0.9905 $ +Y/-Z Source
102 RCC 0 4.45 2.807 0 0 3.680 1.1735
103 RCC 0 4.45 -2.403 0 0 -3.680 1.1735
104 RCC 0 4.45 0 0 0 2.807 1.1735
105 RCC 0 4.45 0 0 0 -2.403 1.1735
106 RCC 0 4.45 -6.159 0 0 13.122 1.186
107 RCC 0 4.45 -6.952 0 0 13.914 1.3385
c
108 RCC 0 -4.45 2.959 0 0 2.972 0.9905   $ -Y/+Z Source
109 RCC 0 -4.45 -2.959 0 0 -2.972 0.9905 $ -Y/-Z Source
110 RCC 0 -4.45 2.807 0 0 3.680 1.1735
111 RCC 0 -4.45 -2.403 0 0 -3.680 1.1735
112 RCC 0 -4.45 0 0 0 2.807 1.1735
113 RCC 0 -4.45 0 0 0 -2.403 1.1735
114 RCC 0 -4.45 -6.159 0 0 13.122 1.186
115 RCC 0 -4.45 -6.952 0 0 13.914 1.3385
c
c Modelling point where tungsten meets the source rod
c Interior
116 TRC 0 4.45 7.312 0 0 -0.076 1.2625 1.186 
117 RCC 0 4.45 7.312 0 0 0.318 1.4455
118 RCC 0 4.45 7.63 0 0 0.889 1.3145
119 RCC 0 4.45 6.487 0 0 0.749 1.186
c
120 TRC 0 -4.45 7.312 0 0 -0.076 1.2625 1.186 
121 RCC 0 -4.45 7.312 0 0 0.318 1.4455
122 RCC 0 -4.45 7.63 0 0 0.889 1.3145
123 RCC 0 -4.45 6.487 0 0 0.749 1.186
c Exterior
124 TRC 0 4.45 7.122 0 0 -0.16 1.778 1.3385
125 RCC 0 4.45 7.122 0 0 1.397 1.778
c
126 TRC 0 -4.45 7.122 0 0 -0.16 1.778 1.3385 
127 RCC 0 -4.45 7.122 0 0 1.397 1.778
c
c Bottom of Source Rod
128 TRC 0 4.45 -7.112 0 0 0.16 1.778 1.3385
129 RCC 0 4.45 -7.112 0 0 -0.318 1.778
130 RCC 0 4.45 -7.43 0 0 -1.27 0.635
131 RCC 0 4.45 -7.34 0 0 -1.27 1.778
c
132 TRC 0 -4.45 -7.112 0 0 0.16 1.778 1.3385
133 RCC 0 -4.45 -7.112 0 0 -0.318 1.778
134 RCC 0 -4.45 -7.43 0 0 -1.27 0.635
135 RCC 0 -4.45 -7.34 0 0 -1.27 1.778
c
c Tungsten Rods
c Lower 
136 RCC 0 4.45 6.487 0 0 0.876 1.1225
137 RCC 0 4.45 7.363 0 0 0.267 1.4265
138 RCC 0 4.45 7.63 0 0 0.889 1.3145
139 RCC 0 4.45 8.519 0 0 11.991 1.829
140 RCC 0 4.45 20.51 0 0 3.81 0.9525
c
141 RCC 0 -4.45 6.487 0 0 0.876 1.1225
142 RCC 0 -4.45 7.363 0 0 0.267 1.4265
143 RCC 0 -4.45 7.63 0 0 0.889 1.3145
144 RCC 0 -4.45 8.519 0 0 11.991 1.829
145 RCC 0 -4.45 20.51 0 0 3.81 0.9525
c Bearing
146 RCC 0 4.45 20.51 0 0 1.27 1.9175
147 RCC 0 -4.45 20.51 0 0 1.27 1.9175
c Upper 
148 RCC 0 4.45 21.78 0 0 3.81 1.829
149 RCC 0 4.45 25.59 0 0 20.955 1.829
150 RCC 0 4.45 46.545 0 0 5.715 1.1115
c
151 RCC 0 -4.45 21.78 0 0 3.81 1.829
152 RCC 0 -4.45 25.59 0 0 20.955 1.829
153 RCC 0 -4.45 46.545 0 0 5.715 1.1115
c -------------------------------------------------------------------
c       Air Cylinder Tower
c -------------------------------------------------------------------
c -------------------------------------------------------------------
c       Primary Shield
c -------------------------------------------------------------------
200 RCC 0 4.45 -55.055 0 0 104.776 1.918
201 RCC 0 4.45 -55.36 0 0 105.081 2.223
202 RCC 0 -4.45 -55.055 0 0 104.776 1.918
203 RCC 0 -4.45 -55.36 0 0 105.081 2.223
c
c Beam Chamber
204 RPP -2.223 29.21 -13.335 13.335 -14.605 14.605
205 RPP -3.35 29.21 -13.97 13.97 -15.24 15.24
c For the Planes Inner_Y= 1.4475+W, Outer_Y=1.744+W
c P X1 Y1 Z1  X2 Y2 Z2  X3 Y3 Z3
206 P 7.99 13.335 14.605 -2.223 5.8975 14.605 -2.223 5.8975 -14.605
207 P 7.99 -13.335 14.605 -2.223 -5.8975 14.605 -2.223 -5.8975 -14.605
208 P 7.99 13.97 15.24 -2.858 6.194 15.24 -2.858 6.194 -15.24
209 P 7.99 -13.97 15.24 -2.858 -6.194 15.24 -2.858 -6.194 -15.24
c
c Lower Section
c Lead
210 RPP -32.703 32.068 -41.91 41.91 -83.934 -44.133
211 P 32.068 6.105 -44.133 32.068 6.105 -83.934 28.352 41.91 -83.934
212 P 32.068 -6.105 -44.133 32.068 -6.105 -83.934 28.352 -41.91 -83.934
213 P -5.011 41.91 -44.133 -5.011 41.91 -83.934 -22.543 28.05 -83.934
214 P -5.011 -41.91 -44.133 -5.011 -41.91 -83.934 -22.543 -28.05 -83.934
215 P -22.543 28.05 -44.133 -32.703 14.191 -44.133 -32.703 14.191 -83.934
216 P -22.543 -28.05 -44.133 -32.703 -14.191 -44.133 -32.703 -14.191 -83.934
c
c Stainless Steel
217 RPP -33.655 33.020 -42.8625 42.8625 -86.475 -44.133
218 P 33.02 6.154 -44.133 33.02 6.154 -86.475 29.21 42.8625 -86.475
219 P 33.02 -6.154 -44.133 33.02 -6.154 -86.475 29.21 -42.8625 -86.475
220 P -5.267 42.8625 -44.133 -23.24 28.654 -44.133 -23.24 28.654 -86.475
221 P -5.267 -42.8625 -44.133 -23.24 -28.654 -44.133 -23.24 -28.654 -86.475
222 P -23.24 28.654 -44.133 -23.24 28.654 -86.475 -33.655 14.446 -86.475
223 P -23.24 -28.654 -44.133 -23.24 -28.654 -86.475 -33.655 -14.446 -86.475
224 RPP 28.575 33.02 -42.8625 42.8625 -44.133 -43.498
c
c Upper Section
c Lead
225 RPP -32.703 28.575 -41.91 41.91 -44.133 45.276
c Steel
226 RPP -33.655 29.21 -42.8625 42.8625 -44.133 49.721
c -------------------------------------------------------------------
c       Secondary Shield        Sem is set at 1mm gap
c -------------------------------------------------------------------
c Lead Cone
250 RPP 29.945 79.464 -41.962 41.962 -42.863 48.724
251 P 29.576 41.962 48.724 29.576 -41.962 48.724 79.464 24.13 25.4 $ Top
252 P 33.006 41.962 -42.863 33.006 -41.962 -42.863 79.464 24.13 -25.4 $ Bottom
253 P 29.576 41.962 48.724 29.576 41.962 -42.863 79.464 24.13 -25.4 $ Side
254 P 29.576 -41.962 48.724 29.576 -41.962 -42.863 79.464 -24.13 -25.4 $ Side
c
255 RPP 79.464 112.315 -24.13 24.13 -25.4 25.4
c Steel Shell
256 RPP 29.31 80.11 -47.9425 47.9425 -43.498 54.801
257 P 30.58 -42.41 49.128 30.58 42.41 49.128 80.11 24.765 26.035 $ Top
258 P 33.12 -42.41 -43.498 33.12 42.41 -43.498 80.11 24.765 -26.035 $ Top
259 P 30.58 -42.41 49.128 30.58 -42.14 -43.498 80.11 -24.765 26.035 $ Side
260 P 30.58 42.41 49.128 30.58 42.14 -43.498 80.11 24.765 26.035 $ Side
c
261 RPP 80.11 113.13 -24.765 24.765 -26.035 26.035
c
c Beam Chamber
265 RPP 29.31 107.415 -13.335 13.335 -14.605 14.605
266 RPP 29.31 107.415 -13.97 13.97 -15.24 15.24
c 
c Cable Ports
270 RCC 38.2 13.335 0 0 48 0 4.445  $ Lead
271 RCC 38.2 13.335 0 0 40 0 3.81  $ Lead
272 RCC 86 0 14.605 0 0 12.79 4.445
273 RCC 86 0 14.605 0 0 12.79 3.810
c
c Side Door Port
275 RPP 40.295 71.855 -17.78 -13.335 -10.795 10.795
276 RPP 40.74 71.22 -18.415 -13.335 -10.16 10.16
277 P 41.502 -13.97 -10.795 41.502 -13.97 10.795 40.295 -17.78 10.795
278 P 43.039 -13.335 -10.16 43.039 -13.335 10.16 40.74 -18.415 -10.16
c
279 RPP 36.485 75.665 -21.59 -17.78 -14.605 14.605
280 RPP 36.93 75.03 -23.007 -18.415 -13.97 13.97
281 P 37.872 -17.78 -14.605 37.872 -17.78 14.605 36.485 -21.59 14.605
282 P 38.317 -18.415 -13.97 38.317 -18.415 13.97 36.485 -23.007 13.97
c
283 RPP 32.305 79.295 -45.505 -21.59 -18.415 18.415
284 RPP 33.12 78.66 -45.505 -23.007 -17.78 17.78
c
c Front Door Port
290 RPP 106.78 113.13 -21.59 21.59 -22.86 22.86
291 RPP  107.415 113.13 -20.955 20.955 -22.225 22.225
c Cable Pass
292 RPP 97.55 113.13 -4.445 4.445 -25.7 -15.54
293 RPP 97.846 113.13 -3.81 3.81 -26.035 -14.605
294 P 97.55 -4.445 -15.54 97.55 4.445 -15.54 106.075 -4.445 -25.7 
295 P 97.846 -3.81 -14.605 97.846 3.81 -14.605 107.415 3.81 -26.035
296 RPP 106.78 113.13 -3.81 3.81 -26.035 -14.605
c -------------------------------------------------------------------
c       Pedestal
c -------------------------------------------------------------------
300 RCC 75.062 0 -86.475 0 0 59.169 13.018
301 RCC 75.062 0 -86.475 0 0 59.169 13.653
302 RCC 75.062 0 -86.475 0 0 -1.27 20.320
c -------------------------------------------------------------------
c       Side Door $ Seam is 0.25 in 0.318 cm
c -------------------------------------------------------------------
310 RPP 41.427 70.192 -19.368 -13.97 -9.207 9.207
311 RPP 40.425 70.827 -19.368 -13.335 -9.8425 9.8425
312 P 43.392 -13.97 -9.207 43.392 -13.97 9.207 41.427 -19.368 -9.207
313 P 42.947 -13.335 -9.8425 42.947 -13.335 9.8425 40.983 -18.733 9.8425
c
314 RPP 37.617 74.003 -22.86 -19.368 -13.0175 13.0175
315 RPP 36.982 74.638 -22.86 -18.733 -13.6525 13.6525
316 P 38.888 -19.368 -13.0175 38.888 -19.368 13.0175 37.437 -22.86 13.0175
317 P 38.444 -18.733 -13.6525 38.444 -18.733 13.6525 37.173 -22.225 13.6525
c
318 RPP 33.998 77.805 -40.492 -22.86 -16.8275 16.8275
319 RPP 33.363 78.448 -41.392 -22.225 -17.4625 17.4625
c -------------------------------------------------------------------
c Front Door
c -------------------------------------------------------------------
320 RPP 102.26 108.007 -12.4565 12.4565 -13.6525 13.6525
321 RPP 101.625 108.007 -13.0915 13.0915 -14.2875 14.2875
322 P 102.26 -10.99 -13.6525 102.26 -10.99 13.6525 108.007 -12.4565 13.6525
323 P 101.625 -11.486 -14.2875 101.625 -11.486 14.2875 107.34 -13.0915 14.2875
c
324 RPP 108.007 115.566 -20.0765 20.0765 -21.275 21.275
325 RPP 107.34 115.566 -20.7115 20.7115 -21.9075 21.9075
326 P 108.007 -18.619 -21.275 108.007 -18.619 21.275 115.566 -20.0675 -21.275
327 P 107.34 -19.254 -21.9075 107.34 -19.254 21.9075 113.055 -20.638 -21.9075
c
328 RPP 115.566 124.517 -24.13 24.13 -35.563 25.403
329 RPP 113.055 125.12 -24.765 24.765 -36.1955 26.0355
c -------------------------------------------------------------------
c       Cable Baffle
c -------------------------------------------------------------------
340 RPP 100.452 106.802 -10.795 24.13 -35.59 -26.67
341 RPP 106.802 112.495 17.78 24.13 -35.59 -26.67
342 RPP 106.802 112.495 -10.795 -4.445 -35.59 -26.67
343 RPP 106.802 112.495 -4.445 8.89 -35.59 -31.75
c
344 RPP 99.817 107.437 -11.43 24.765 -36.195 -26.035
345 RPP 107.437 113.13 17.415 24.765 -36.195 -26.035
346 RPP 107.437 113.13 -11.43 -3.81 -36.195 -26.035
347 RPP 107.437 113.13 -3.81 9.525 -36.195 -31.115
c -------------------------------------------------------------------
c       Port Plugs
c -------------------------------------------------------------------
350 RCC  86 0 14.605 0 0 11.43 3.759
351 RCC  86 0 26.035 0 0 2.54 5.08
352 RCC  38.2 13.335 0 0 27.699 0 3.759 $ Use plane 260 for the diagonal
353 RCC  38.2 39.6954 0 0.824 2.54 0 5.08
c -------------------------------------------------------------------
c       Kill Volume
c -------------------------------------------------------------------
999 RPP -100 200 -100 100 -100 100      $ Kill Volume

c -------------------------------------------------------------------
c
c                       Data Cards
c
c -------------------------------------------------------------------
MODE P
NPS 3.7E10
PRINT 10 60 110
PRDMP J 5e7 0 1
RAND 97008386995783
IMP:P 1 32R 0
c -------------------------------------------------------------------
c           Material Cards
c -------------------------------------------------------------------
m1 27000 -1         $ Nat-Co
c
c   
m100 $$ Stainless Steel 316, p=8.0 PNNL-15870
        6000 -0.0008
        25055 -0.02
        15031 -0.00045
        16000 -0.0003
        14000 -0.01
        24000 -0.17
        28000 -0.12
        42000 -0.025
        26000 -0.65345
m200 $$ Tungsten, p=19.3 
        74000 -1.0
m300 $$ Bronze, p=8.4 PNNL-15870
        13027 -0.028528
        14000 -0.003339
        25055 -0.003555
        26000 -0.010208
        28000 -0.006718
        29000 -0.874155
        30000 -0.036037
        50000 -0.024503
        82000 -0.012957
c
m400 $$ Lead, p=11.35
        82000  -1.0
c
m99 $ Dry Air, p=0.001205 PNNL-15870
        6000 -0.000124
        7014 -0.752316
        7015 -0.002944
        8016 -0.231153
        8017 -0.000094
        8018 -0.000535
        18000 -0.012827
c -------------------------------------------------------------------
c       SDEF Card
c -------------------------------------------------------------------
SDEF PAR=2
     ERG=D1
     POS=D2
     AXS=0 0 1
     RAD=D3
     EXT=D4
c
c
SI1 L 1.1732  1.3325	$ Cobalt-60 gammas
SP1 D 0.49969 0.50031
c
SI2 L 0 4.45 4.445  0 -4.45 4.445  0 4.45 -4.445  0 -4.45 -4.445
SP2 0.25 0.25 0.25 0.25		$ Seting Center of Source Points
c
SI3 L 0 0.9906	 $ Isotropic Radial Sampling
SP3 -21 1
c
SI4 L -1.4859 1.4869	$ Isotopic Axial Sampling	
SP4 -21 0
c c ------------------------------------------------------------------------
c c                   Variance Reduction
c c ------------------------------------------------------------------------
F9994:P 998
FM9994 6.39e+11 $ 23629.9 Ci * 3.7E10 Bq/Ci * 2 photons per disintegration * 3600 sec/hr * 1E-7 mrem/pSv   
WWG 9994 0 0 $ Last zero means half the avg source weight used for lower window bound
c
MESH GEOM=xyz REF=0 0 0 ORIGIN=-100 -100 -100  
     IMESH = -32 0 103 130 200
     IINTS = 2 32 50 30 2
     JMESH = -45 45 100
     JINTS = 2 100 2
     KMESH = -45 52 100 
     KINTS = 2 110 5
c
WWP:P 5 3 5 0 -1 J J J
c ------------------------------------------------------------------------
c                   Dose Response
c ------------------------------------------------------------------------
c Photon Flux-to-Dose Rate Conversion Factors $ [pSv-cm^2]
c Extracted from ICRP 116 Dose Conversion Coefficients
DE0 1.00E-02 1.50E-02 2.00E-02 3.00E-02 4.00E-02 5.00E-02
     6.00E-02 7.00E-02 8.00E-02 1.00E-01 1.50E-01 2.00E-01
     3.00E-01 4.00E-01 5.00E-01 5.11E-01 6.00E-01 6.62E-01
     8.00E-01 1.00E+00 1.12E+00 1.33E+00 1.50E+00 2.00E+00
     3.00E+00 4.00E+00 5.00E+00 6.00E+00 6.13E+00 8.00E+00
     1.00E+01 1.50E+01 2.00E+01 3.00E+01 4.00E+01 5.00E+01
     6.00E+01 8.00E+01 1.00E+02 1.50E+02 2.00E+02 3.00E+02
     4.00E+02 5.00E+02 6.00E+02 8.00E+02 1.00E+03 1.50E+03
     2.00E+03 3.00E+03 4.00E+03 5.00E+03 6.00E+03 8.00E+03
     1.00E+04
DF0 0.0337 0.0664 0.0986 0.1580 0.1990 0.2260 0.2480
     0.2730 0.2970 0.3550 0.5280 0.7210 1.1200 1.5200
     1.9200 1.9600 2.3000 2.5400 3.0400 3.7200 4.1000
     4.7500 5.2400 6.5500 8.8400 10.8000 12.7000 14.4000
     14.6000 17.6000 20.6000 27.7000 34.4000 46.1000 56.0000
     64.4000 71.2000 82.0000 89.7000 102.0000 111.0000 121.0000
     128.0000 133.0000 136.0000 142.0000 145.0000 152.0000 156.0000
     161.0000 165.0000 168.0000 170.0000 172.0000 175.0000
c ------------------------------------------------------------------------
c                   Mesh Tallies
c ------------------------------------------------------------------------
FMESH14:p GEOM=XYZ Origin=-2.223 -13.335 -14.605
        IMESH=107.415    IINTS=110
        JMESH=13.335    JINTS=26
        KMESH=14.605    KINTS=28
FC14 FMESH Tally over the entire Span of the beam Chamber
FM14 6.29499e+11
c
FMESH24:p GEOM=XYZ ORIGIN=-100 -100 -100
        IMESH=200       IINTS=300
        JMESH=100       JINTS=200
        KMESH=100       KINTS=200
FC24 FMESH Tally over the Whole Volume
FM24 6.29499e+11 
c ------------------------------------------------------------------------
c                   Point Detector Tallies
c ------------------------------------------------------------------------
F115:p 29.31 -48 56 1.0  29.31 -28 56 1.0  29.31 0 56 1.0  29.31 28 56 1.0  29.31 48 56 1.0
FC115 Point Detector Tallies above the Seam -Y to +Y
FM115 6.29499e+11
c
F125: 29.31 -49 56 1.0  29.31 -49 25 1.0  29.31 -49 0 1.0 29.31 -49 -25 1.0  29.31 -49 -56 1.0
FC125 Point Detector Tallies -Y of the Seam Vertically
FM125 6.29499e+11
c
F135:p 29.31 -48 -45 1.0  29.31 -28 -45 1.0  29.31 0 -45 1.0  29.31 28 -45 1.0  29.31 48 -45 1.0
FC135 Point Detector Tallies above the Seam -Y to +Y
FM135 6.29499e+11
c
F145:p 29.31 49 56 1.0  29.31 49 25 1.0  29.31 49 0 1.0 29.31 49 -25 1.0  29.31 49 -56 1.0
FC145 Point Detector Tallies -Y of the Seam Vertically
FM145 6.29499e+11
c
F155:p 108 -5 -38 2.0
FC155 Point Detector at Cable Baffle Outlet
FM155 6.29499e+11
c
F165:p 0 4.45 51 1.0  0 -4.45 51 1.0
FC165 Point Detectors directly above the Sources
FM165 6.29499e+11
