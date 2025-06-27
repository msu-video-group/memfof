getSpringRecords = () => {
    return [
        {
            x: 2.09,
            y: 3.289,
            runtime: 472.0,
            label: "MEMFOF",
            tags: ["multi-frame"]
        },
        {
            x: 8.08,
            y: 4.482,
            runtime: 885.0,
            label: "MemFlow",
            tags: ["multi-frame"]
        },
        {
            x: 18.97,
            y: 4.152,
            runtime: 1403.0,
            label: "StreamFlow",
            tags: ["multi-frame"]
        },
        {
            x: 2.01,
            y: 4.565,
            runtime: 6524.0,
            label: "CrocoFlow",
            tags: ["two-frame"]
        },
        {
            x: 8.19,
            y: 3.686,
            runtime: 286.0,
            label: "SEA-RAFT (M)",
            tags: ["two-frame"]
        },
        {
            x: 4.16,
            y: 6.710,
            runtime: 167.0,
            label: "FlowNet2 (w/o ft)",
            tags: ["two-frame"]
        },
        {
            x: 7.97,
            y: 6.790,
            runtime: 557.0,
            label: "RAFT (w/o ft)",
            tags: ["two-frame"]
        },
        {
            x: 13.26,
            y: 7.074,
            runtime: 1185.0,
            label: "GMA (w/o ft)",
            tags: ["two-frame"]
        },
        {
            x: 8.49,
            y: 4.809,
            runtime: 295.0,
            label: "RPKNet (w/o ft)",
            tags: ["two-frame"]
        },
        {
            x: 8.08,
            y: 5.759,
            runtime: 885.0,
            label: "MemFlow (w/o ft)",
            tags: ["multi-frame"]
        },
        {
            x: 18.97,
            y: 5.215,
            runtime: 1403.0,
            label: "StreamFlow (w/o ft)",
            tags: ["multi-frame"]
        },
        {
            x: 2.09,
            y: 3.600,
            runtime: 472.0,
            label: "MEMFOF (w/o ft)",
            tags: ["multi-frame"]
        }
    ]

};

document.addEventListener("DOMContentLoaded", function () {
    const ctx = document.getElementById('spring-chart');

    new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'Two-frame',
                data: getSpringRecords().filter(item => item.tags.includes("two-frame")),
                backgroundColor: 'rgba(54, 162, 235, 0.7)',
                borderColor: 'rgba(54, 162, 235, 1)',
                pointRadius: 6,
                pointHoverRadius: 8
            },
            {
                label: 'Multi-frame',
                data: getSpringRecords().filter(item => item.tags.includes("multi-frame")),
                backgroundColor: 'rgba(235, 99, 132, 0.7)',
                borderColor: 'rgba(235, 99, 132, 1)',
                pointRadius: 6,
                pointHoverRadius: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    labels: {
                        color: '#444',
                        font: {
                            size: 14
                        }
                    }
                },
                tooltip: {
                    callbacks: {
                        label: context => `1px: ${context.parsed.y.toFixed(2)}%, VRAM@1080p: ${context.parsed.x.toFixed(2)} GB, Runtime: ${context.raw.runtime} ms`
                    }
                },
                datalabels: {
                    align: 'right',
                    offset: 10,
                    font: context => {
                        const label = context.dataset.data[context.dataIndex].label;
                        return {
                            weight: (label === 'MEMFOF' || label === 'MEMFOF (w/o ft)') ? 'bold' : 'normal',
                            size: 12
                        };
                    },
                    formatter: value => value.label
                },
                betterDirectionArrow: {}
            },
            scales: {
                x: {
                    type: 'linear',
                    position: 'bottom',
                    title: {
                        display: true,
                        text: 'Memory Usage at 1080p (GB)',
                        font: { size: 16 }
                    },
                    beginAtZero: true,
                    max: 24
                },
                y: {
                    title: {
                        display: true,
                        text: '1px Error (%)',
                        font: { size: 16 }
                    },
                    beginAtZero: true,
                }
            },
            layout: {
                padding: {
                right: 40
                }
            },
        },
        plugins: [ChartDataLabels, betterDirectionPlugin]
    });
});