#!/bin/bash
# ChaosBF Quick Start Script

echo "========================================"
echo "ChaosBF Quick Start"
echo "========================================"
echo ""

cd /home/ubuntu/chaosbf

echo "1. Running a simple example..."
echo "   Code: ++[>+<-]."
python3 src/chaosbf.py "++[>+<-]." --energy 100 --steps 100
echo ""

echo "2. Running the hot seed program..."
echo "   Code: ++[>+<-].:{;}{?}^*=@=.#%{?}{?}v=.#"
python3 src/chaosbf.py "++[>+<-].:{;}{?}^*=@=.#%" --energy 180 --temp 0.6 --steps 500
echo ""

echo "3. Generating visualizations..."
python3 src/visualize.py "++[>+<-].:{;}{?}^*=@=.#%" --steps 1000 --output output/quickstart
echo ""

echo "========================================"
echo "Quick start complete!"
echo "========================================"
echo ""
echo "Generated files:"
ls -lh output/quickstart* 2>/dev/null || echo "  (No files generated)"
echo ""
echo "Next steps:"
echo "  - Explore examples: python3 examples/seed_programs.py all"
echo "  - Read the guide: cat docs/guide.md"
echo "  - Create your own programs!"
