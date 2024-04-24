@ PATH C:\MCNP62\MCNP_CODE\bin;%PATH%
@ set DATAPATH=C:\MCNP62\MCNP_DATA
@ set DISPLAY=localhost: 0
chdir C:\Work\Project-3887-main\MCNP-Inputs\Server_Run
mcnp6 name = JLS484_V10_Slave0.i tasks 26
mcnp6 name = JLS484_V10_Slave1.i wwinp=JLS484_V10_Slave0.ie tasks 26
mcnp6 name = JLS484_V10_Slave2.i wwinp=JLS484_V10_Slave1.ie tasks 26
mcnp6 name = JLS484_V10_Slave3.i wwinp=JLS484_V10_Slave2.ie tasks 26