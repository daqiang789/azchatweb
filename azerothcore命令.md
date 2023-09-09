### 1、Azerothcore安装指南
https://www.azerothcore.org/wiki/installation

### 2、数据库&依赖安装：
sudo apt update && sudo apt full-upgrade -y && sudo apt install git cmake make gcc g++ clang libssl-dev libbz2-dev libreadline-dev libncurses-dev libboost-all-dev mariadb-server mariadb-client libmariadb-dev libmariadb-dev-compat

### 3、基础软件安装：
debian/ubuntu
sudo apt update && sudo apt install git curl unzip sudo

### 4、下载服务器源代码：
git clone https://github.com/trickerer/AzerothCore-wotlk-with-NPCBots.git; cd AzerothCore-wotlk-with-NPCBots

### 5、安装构建服务器所需的依赖：
./acore.sh install-deps

### 6、开始构建服务端软件：
./acore.sh compiler all

### 7、升级一下数据库：
apt update
curl -LsS https://r.mariadb.com/downloads/mariadb_repo_setup | sudo bash
sudo apt install mariadb-server -y

### 8、设置数据库-创建服务端专用的帐户
sudo mysql -u root
DROP USER IF EXISTS 'acore'@'localhost';
DROP USER IF EXISTS 'acore'@'127.0.0.1';
CREATE USER 'acore'@'localhost' IDENTIFIED BY 'acore';
CREATE USER 'acore'@'127.0.0.1' IDENTIFIED BY 'acore';
GRANT ALL PRIVILEGES ON * . * TO 'acore'@'localhost' WITH GRANT OPTION;
GRANT ALL PRIVILEGES ON * . * TO 'acore'@'127.0.0.1' WITH GRANT OPTION;
FLUSH PRIVILEGES;
exit;

### 9、下载客户端资料：
./acore.sh client-data

### 10、复制服务器默认配置文件，
cp env/dist/etc/authserver.conf.dist env/dist/etc/authserver.conf
cp env/dist/etc/worldserver.conf.dist env/dist/etc/worldserver.conf

### 11、修改机器人上限
vim env/dist/etc/worldserver.conf
（将bot改成NpcBot.MaxBots = 39）

### 12、首次运行世界服务
./acore.sh run-worldserver

### 13、开启数据库远程访问权限
mysql -u root
GRANT ALL PRIVILEGES ON *.* TO 'acore'@'%' IDENTIFIED BY 'acore' WITH GRANT OPTION;
FLUSH PRIVILEGES;
exit;
vim /etc/mysql/mariadb.conf.d/50-server.cnf
bind-address=0.0.0.0
systemctl restart mariadb

### 14、服务端开放远程访问权限
mysql -u root
use acore_auth;
UPDATE realmlist SET address = '192.168.110.235' WHERE id = 1;
exit;

### 15 运行所有服务
./acore.sh run-worldserver
./acore.sh run-authserver

### 16、创建帐号（世界服务中输入命令）
account create 帐号 密码
设置为管理员
account set gmlevel 帐号 3 -1
修改密码
account set password 帐号 密码 密码2

### 其它
NPCbot插件：
https://github.com/NetherstormX/NetherBot

添加机器人
npcbot spawn
