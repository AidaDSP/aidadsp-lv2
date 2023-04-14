function (event) {
    function input_size_changed(input_size_f) {
        var input_size = parseInt(input_size_f);
        /* TODO match HTML elements
        switch (input_size) {
        case 0:
        case 1:
            event.icon.find('.conditioned-param-1').hide();
            event.icon.find('.conditioned-param-2').hide();
            break
        case 2:
            event.icon.find('.conditioned-param-1').show();
            event.icon.find('.conditioned-param-2').hide();
            break
        case 3:
            event.icon.find('.conditioned-param-1').show();
            event.icon.find('.conditioned-param-2').show();
            break
        }
        */
    }

    if (event.type === 'start') {
        for (var i in event.ports) {
            if (event.ports[i].symbol === 'ModelInSize') {
                input_size_changed(event.ports[i].value);
                break;
            }
        }
    }
    else if (event.symbol === 'ModelInSize') {
        input_size_changed(event.value);
    }
}
