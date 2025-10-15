is_apper_trading = True

real_app_key = ""
real_app_secret = ""

paper_app_key = "E0UCpiyUiLivuR73bpg0LKkgN2773yYjUw9GZBa4r_o"
paper_app_secret ="pe5AVGat43XGCDxTrxj7HcdFqRFC99FDI2WKYB4MJIs"

real_host_url = "https://api.kiwoom.com"
paper_host_url = "https://mockapi.kiwoom.com"

real_socket_url = "wss://api.kiwoom.com:10000"
paper_socket_url = "wss://mockapi.kiwoom.com:10000"

app_key = paper_app_key if is_apper_trading else real_app_key
app_secret = paper_app_secret if is_apper_trading else real_app_secret
host_url = paper_host_url if is_apper_trading else real_host_url
socket_url = paper_socket_url if is_apper_trading else real_socket_url