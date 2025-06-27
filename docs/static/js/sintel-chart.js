getRecords = () => {
    return [
        {
            sintel_clean: 3.86,
            sintel_final: 5.04,
            memory_1080p: "1.41 GB",
            label: "PWC-Net",
        },
        {
            sintel_clean: 4.16,
            sintel_final: 5.74,
            memory_1080p: "4.16 GB",
            label: "FlowNet2",
        },
        {
            sintel_clean: 2.238,
            sintel_final: 3.806,
            memory_1080p: "1.34 GB",
            label: "Flow1D",
        },
        {
            sintel_clean: 2.054,
            sintel_final: 3.090,
            memory_1080p: "1.32 GB",
            label: "MeFlow",
        },
        {
            sintel_clean: 1.609,
            sintel_final: 2.855,
            memory_1080p: "7.97 GB",
            label: "RAFT",
        },
        {
            sintel_clean: 1.388,
            sintel_final: 2.470,
            memory_1080p: "13.26 GB",
            label: "GMA",
        },
        {
            sintel_clean: 1.442,
            sintel_final: 2.865,
            memory_1080p: "8.19 GB",
            label: "SEA-RAFT (M)",
        },
        {
            sintel_clean: 1.309,
            sintel_final: 2.601,
            memory_1080p: "8.22 GB",
            label: "SEA-RAFT (L)",
        },
        {
            sintel_clean: 1.159,
            sintel_final: 2.088,
            memory_1080p: "OOM",
            label: "FlowFormer",
        },
        {
            sintel_clean: 1.315,
            sintel_final: 2.657,
            memory_1080p: "8.49 GB",
            label: "RPKNet",
        },
        {
            sintel_clean: 1.092,
            sintel_final: 2.436,
            memory_1080p: "2.01 GB",
            label: "CrocoFlow",
        },
        // {
        //     sintel_clean: 1.754,
        //     sintel_final: 2.475,
        //     memory_1080p: "N/A",
        //     label: "DDVM",
        // },
        {
            sintel_clean: 1.041,
            sintel_final: 1.874,
            memory_1080p: "18.97 GB",
            label: "StreamFlow",
        },
        {
            sintel_clean: 1.046,
            sintel_final: 1.914,
            memory_1080p: "8.08 GB",
            label: "MemFlow",
        },
        // {
        //     sintel_clean: 1.081,
        //     sintel_final: 1.840,
        //     memory_1080p: "N/A",
        //     label: "MemFlow-T",
        // },
        {
            sintel_clean: 1.005,
            sintel_final: 1.713,
            memory_1080p: "17.74 GB",
            label: "VideoFlow-BOF",
        },
        {
            sintel_clean: 0.963,
            sintel_final: 1.907,
            memory_1080p: "2.09 GB",
            label: "MEMFOF",
        },
        {
            sintel_clean: 0.991,
            sintel_final: 1.649,
            memory_1080p: "OOM",
            label: "VideoFlow-MOF",
        },
    ]

};

document.addEventListener("DOMContentLoaded", function () {
    const ctx = document.getElementById("sintel-chart");
    const records = getRecords();

    const uniqueMethods = [...new Set(records.map(d => d.label))];

    const grouped = uniqueMethods.map(label => {
        return records.find(d => d.label === label);
    });

    grouped.sort((a, b) => a.sintel_clean - b.sintel_clean);

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: grouped.map(d => d.label),
            datasets: [
                {
                    label: 'Sintel Clean',
                    data: grouped.map(d => ({ x: d.label, y: d.sintel_clean, memory_1080p: d.memory_1080p })),
                    backgroundColor: 'rgba(235, 99, 132, 0.7)',
                    borderColor: 'rgba(235, 99, 132, 1)',
                    borderWidth: 1
                },
                {
                    label: 'Sintel Final',
                    data: grouped.map(d => ({ x: d.label, y: d.sintel_final, memory_1080p: d.memory_1080p })),
                    backgroundColor: 'rgba(54, 162, 235, 0.7)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: {
                        font: { size: 14 }
                    }
                },
                tooltip: {
                    callbacks: {
                        label: context => `EPE: ${context.parsed.y.toFixed(2)}px, VRAM@1080p: ${context.raw.memory_1080p}`
                    }
                },
                betterDirectionArrow: {}
            },
            scales: {
                x: {
                    ticks: {
                        font: function (context) {
                            return {
                                weight: context.tick.label === 'MEMFOF' ? 'bold' : 'normal',
                            };
                        }
                    },
                },
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'EndPoint Error (px)',
                        font: { size: 16 }
                    }
                },
            },
            layout: {
                padding: {
                right: 40
                }
            },
        },
        plugins: [betterDirectionPlugin]
    });
});
