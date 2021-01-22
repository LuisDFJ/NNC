FileNameMain = main
FileNameLib1 = DNN/DNN
FileNameLib2 = CNN/CNN
FileNameLib3 = TOP/TOP
FileNameCom1 = common/common
FileNameCom2 = common/Network
INCLUDE = $(PWD)/include/
DEBUG = $(PWD)/debug/
CC = gcc
CFLAGS = -Wall -I$(INCLUDE)
CH = chmod a+x


all: $(DEBUG)$(FileNameMain)
$(DEBUG)$(FileNameMain) : $(FileNameMain).c $(INCLUDE)$(FileNameLib1).c $(INCLUDE)$(FileNameLib2).c $(INCLUDE)$(FileNameLib3).c $(INCLUDE)$(FileNameCom1).c $(INCLUDE)$(FileNameCom2).c 
	@echo COMPILING!
	@$(CC) $(CFLAGS) -o $@ $^ -lm
	@$(CH) $(DEBUG)$(FileNameMain)
	@echo DONE!
run:
	@echo RUNNING: $(FileNameMain)
	@echo -----------------------
	@$(DEBUG)$(FileNameMain)
	
clean:
	@rm -f $(DEBUG)*
	@echo FILES REMOVED!
