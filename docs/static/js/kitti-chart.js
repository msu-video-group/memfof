getKittiRecords = () => {
    return [
        {
            fl: 9.60,
            is_stereo: false,
            memory_1080p: "1.41 GB",
            label: "PWC-Net",
        },
        {
            fl: 10.41,
            is_stereo: false,
            memory_1080p: "4.16 GB",
            label: "FlowNet2",
        },
        {
            fl: 6.27,
            is_stereo: false,
            memory_1080p: "1.34 GB",
            label: "Flow1D",
        },
        {
            fl: 4.95,
            is_stereo: false,
            memory_1080p: "1.32 GB",
            label: "MeFlow",
        },
        {
            fl: 5.10,
            is_stereo: false,
            memory_1080p: "7.97 GB",
            label: "RAFT",
        },
        {
            fl: 5.15,
            is_stereo: false,
            memory_1080p: "13.26 GB",
            label: "GMA",
        },
        {
            fl: 4.64,
            is_stereo: false,
            memory_1080p: "8.19 GB",
            label: "SEA-RAFT (M)",
        },
        {
            fl: 4.30,
            is_stereo: false,
            memory_1080p: "8.22 GB",
            label: "SEA-RAFT (L)",
        },
        {
            fl: 4.68,
            is_stereo: false,
            memory_1080p: "OOM",
            label: "FlowFormer",
        },
        {
            fl: 4.64,
            is_stereo: false,
            memory_1080p: "8.49 GB",
            label: "RPKNet",
        },
        {
            fl: 3.64,
            is_stereo: false,
            memory_1080p: "2.01 GB",
            label: "CrocoFlow",
        },
        // {
        //     fl: 3.26,
        //     is_stereo: false,
        //     memory_1080p: "N/A",
        //     label: "DDVM",
        // },
        {
            fl: 4.24,
            is_stereo: false,
            memory_1080p: "18.97 GB",
            label: "StreamFlow",
        },
        {
            fl: 4.10,
            is_stereo: false,
            memory_1080p: "8.08 GB",
            label: "MemFlow",
        },
        // {
        //     fl: 3.88,
        //     is_stereo: false,
        //     memory_1080p: "N/A",
        //     label: "MemFlow-T",
        // },
        {
            fl: 4.44,
            is_stereo: false,
            memory_1080p: "17.74 GB",
            label: "VideoFlow-BOF",
        },
        {
            fl: 2.94,
            is_stereo: false,
            memory_1080p: "2.09 GB",
            label: "MEMFOF",
        },
        {
            fl: 3.65,
            is_stereo: false,
            memory_1080p: "OOM",
            label: "VideoFlow-MOF",
        }
    ]

};

document.addEventListener("DOMContentLoaded", function () {
    const ctx = document.getElementById("kitti-chart");
    const records = getKittiRecords();

    const uniqueMethods = [...new Set(records.map(d => d.label))];

    const grouped = uniqueMethods.map(label => {
        return records.find(d => d.label === label);
    });

    grouped.sort((a, b) => a.fl - b.fl);

    const backgroundColors = grouped.map(d =>
        d.is_stereo ? 'rgba(235, 99, 132, 0.7)' : 'rgba(54, 162, 235, 0.7)'
    );

    const borderColors = grouped.map(d =>
        d.is_stereo ? 'rgba(235, 99, 132, 1)' : 'rgba(54, 162, 235, 1)'
    );

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: grouped.map(d => d.label),
            datasets: [
                {
                    data: grouped.map(d => ({ x: d.label, y: d.fl, memory_1080p: d.memory_1080p })),
                    backgroundColor: backgroundColors,
                    borderColor: borderColors,
                    borderWidth: 1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false,
                    labels: {
                        font: { size: 14 }
                    }
                },
                tooltip: {
                    callbacks: {
                        label: context => `Fl-all: ${context.parsed.y.toFixed(2)} %, VRAM@1080p: ${context.raw.memory_1080p}`
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
                        text: 'Fl-all (%)',
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
