all:
	@echo "Building all exercises..."
	@$(MAKE) -C ejercicio1
	@$(MAKE) -C ejercicio2
	@echo "✓ All exercises built successfully"

ejercicio1:
	@$(MAKE) -C ejercicio1

ejercicio2:
	@$(MAKE) -C ejercicio2

run1:
	@$(MAKE) -C ejercicio1 run

run2:
	@$(MAKE) -C ejercicio2 run

plots1:
	@$(MAKE) -C ejercicio1 plots

clean:
	@$(MAKE) -C ejercicio1 clean
	@$(MAKE) -C ejercicio2 clean
	@echo "✓ Cleaned all exercises"

clean1:
	@$(MAKE) -C ejercicio1 clean

clean2:
	@$(MAKE) -C ejercicio2 clean

.PHONY: all ejercicio1 ejercicio2 run1 run2 plots1 clean clean1 clean2
